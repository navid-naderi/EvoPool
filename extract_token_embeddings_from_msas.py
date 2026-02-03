#!/usr/bin/env python3


import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import esm
from transformers import AutoModelForMaskedLM

import os
from pathlib import Path

# Choose a big filesystem on the cluster
HF_ROOT = Path("./hf_cache")
HF_ROOT.mkdir(parents=True, exist_ok=True)

# Hugging Face caches
os.environ["HF_HOME"] = str(HF_ROOT)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_ROOT / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_ROOT / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(HF_ROOT / "datasets")  # optional


def read_a3m(path: Path) -> List[Tuple[str, str]]:
    """
    Minimal A3M parser.
    Returns list of (seq_id, raw_aligned_seq) including gaps/insertions.
    """
    records = []
    with path.open("r") as f:
        seq_id = None
        seq_chunks = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    records.append((seq_id, "".join(seq_chunks)))
                seq_id = line[1:].split()[0]  # token before first space
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if seq_id is not None:
            records.append((seq_id, "".join(seq_chunks)))
    return records


def a3m_clean_keep_insertions(seq: str) -> str:
    """
    Your requested cleaning:
      - uppercase everything (keeps former lowercase insertions)
      - remove gap characters '-' and '.'
      - do NOT remove any other letters
    """
    seq = seq.upper()
    seq = seq.replace("-", "").replace(".", "")
    return seq


def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msa-root", required=True, help="Root folder containing .a3m files.")
    ap.add_argument("--out-root", required=True, help="Where to write embeddings.")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size for ESM inference.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="cuda or cpu (default auto-detect).")
    ap.add_argument("--pattern", default=".a3m", help="File suffix to include (default: .a3m).")
    ap.add_argument("--max-len", type=int, default=1022,
                    help="Max tokens including special tokens; sequences are truncated if needed.")
    ap.add_argument("--limit-msas", type=int, default=None,
                    help="Optional: stop after processing this many MSAs.")
    ap.add_argument("--plm", type=str, default='esm2_8M',
                    help="PLM Type; ESM-2 supported for now")
    args = ap.parse_args()

    msa_root = Path(args.msa_root)
    out_root = Path(args.out_root)
    # out_root = Path(args.out_root + '_{}'.format(args.plm)) # add the PLM name to the output folder
    out_root.mkdir(parents=True, exist_ok=True)

    if 'esm2' in args.plm:

        print(f"Loading ESM2 model on {args.device} …")
        if args.plm == 'esm2_8M':
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        elif args.plm == 'esm2_35M':
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        elif args.plm == 'esm2_150M':
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        elif args.plm == 'esm2_650M':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            raise Exception

        model.eval()
        model = model.to(args.device)
        batch_converter = alphabet.get_batch_converter()
        pad_idx = alphabet.padding_idx
        last_layer = model.num_layers  # take representations from the final layer

        # Collect A3M files
        a3m_files = [p for p in msa_root.rglob(f"*{args.pattern}") if p.is_file()]
        a3m_files.sort()
        if args.limit_msas is not None:
            a3m_files = a3m_files[: args.limit_msas]

        print(f"Found {len(a3m_files)} A3M files.")
        for i, msa_path in enumerate(a3m_files, 1):
            rel = msa_path.relative_to(msa_root)
            msa_out_dir = out_root / rel.parent / (rel.stem + "_embeds")
            msa_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{i}/{len(a3m_files)}] {msa_path} -> {msa_out_dir}")

            # Parse and clean
            records = read_a3m(msa_path)
            clean_records = []
            for sid, aligned in records:
                seq = a3m_clean_keep_insertions(aligned)
                if not seq:
                    continue
                clean_records.append((sid, seq))
            if not clean_records:
                print("  (no valid sequences after cleaning; skipping)")
                continue

            # Prepare batches
            for batch in batched(clean_records, args.batch_size):
                # ESM adds BOS/EOS; cap raw length to max_len-2
                capped = []
                for sid, s in batch:
                    if len(s) > args.max_len - 2:
                        s = s[: args.max_len - 2]
                    capped.append((sid, s))

                # ESM expects (label, seq); labels optional
                labels, strs, toks = batch_converter([("", s) for _, s in capped])
                toks = toks.to(args.device)

                with torch.no_grad():
                    out = model(toks, repr_layers=[last_layer], return_contacts=False)
                reps = out["representations"][last_layer]  # [B, T, C], includes BOS/EOS

                # Save each sequence embedding excluding BOS/EOS and padding
                for bi, (sid, s) in enumerate(capped):
                    tok_row = toks[bi]
                    rep_row = reps[bi]
                    nonpad = (tok_row != pad_idx).sum().item()  # length incl BOS/EOS
                    emb = rep_row[1:nonpad-1].cpu().float().numpy()  # [L, D]

                    safe_sid = re.sub(r"[^A-Za-z0-9._+-]", "_", sid)[:200]
                    out_path = msa_out_dir / f"{safe_sid}.npy"
                    np.save(out_path, emb)

    elif 'esmc' in args.plm:# or 'e1' in args.plm:
    # elif 'esmc' in args.plm or 'e1' in args.plm:
        print(f"Loading {args.plm} model on {args.device} …")
        if args.plm == 'esmc_600M':
            model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_large', trust_remote_code=True)
        else:
            raise Exception

        model.eval()
        model = model.to(args.device)

        # Collect A3M files
        a3m_files = [p for p in msa_root.rglob(f"*{args.pattern}") if p.is_file()]
        a3m_files.sort()
        if args.limit_msas is not None:
            a3m_files = a3m_files[: args.limit_msas]

        print(f"Found {len(a3m_files)} A3M files.")
        for i, msa_path in enumerate(a3m_files, 1):
            rel = msa_path.relative_to(msa_root)
            msa_out_dir = out_root / rel.parent / (rel.stem + "_embeds")
            msa_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{i}/{len(a3m_files)}] {msa_path} -> {msa_out_dir}")

            # Parse and clean
            records = read_a3m(msa_path)
            clean_records = []
            for sid, aligned in records:
                seq = a3m_clean_keep_insertions(aligned)
                if not seq:
                    continue
                clean_records.append((sid, seq))
            if not clean_records:
                print("  (no valid sequences after cleaning; skipping)")
                continue

            # Prepare batches
            for batch in batched(clean_records, args.batch_size):

                # Extract per-residue representations
                all_token_representations = []
                with torch.no_grad():

                    seq_batch = [pair[1] for pair in batch]
                    tokenized = model.tokenizer(seq_batch, padding=True, return_tensors='pt').to(args.device)
                    output = model(**tokenized) # get all hidden states with output_hidden_states=True
                    batch_token_representations = output.last_hidden_state.detach().cpu().numpy() # remove the eos and bos token embeddings

                # Save each sequence embedding
                for bi, (sid, s) in enumerate(batch):
                    li = len(s)
                    emb = batch_token_representations[bi][1:li+1]  # [L, D]


                    safe_sid = re.sub(r"[^A-Za-z0-9._+-]", "_", sid)[:200]
                    out_path = msa_out_dir / f"{safe_sid}.npy"
                    np.save(out_path, emb)

    elif 'e1' in args.plm:

        from E1.batch_preparer import E1BatchPreparer
        from E1.modeling import E1ForMaskedLM

        print(f"Loading {args.plm} model on {args.device} …")
        if args.plm == 'e1_600M':
            model = E1ForMaskedLM.from_pretrained("Profluent-Bio/E1-600m")
        else:
            raise Exception

        model.eval()
        model = model.to(args.device)

        # Collect A3M files
        a3m_files = [p for p in msa_root.rglob(f"*{args.pattern}") if p.is_file()]
        a3m_files.sort()
        if args.limit_msas is not None:
            a3m_files = a3m_files[: args.limit_msas]

        print(f"Found {len(a3m_files)} A3M files.")
        for i, msa_path in enumerate(a3m_files, 1):
            rel = msa_path.relative_to(msa_root)
            msa_out_dir = out_root / rel.parent / (rel.stem + "_embeds")
            msa_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{i}/{len(a3m_files)}] {msa_path} -> {msa_out_dir}")

            # Parse and clean
            records = read_a3m(msa_path)
            clean_records = []
            for sid, aligned in records:
                seq = a3m_clean_keep_insertions(aligned)
                if not seq:
                    continue
                clean_records.append((sid, seq))
            if not clean_records:
                print("  (no valid sequences after cleaning; skipping)")
                continue

            # Prepare batches
            for batch in batched(clean_records, args.batch_size):

                # Extract per-residue representations
                with torch.no_grad():

                    seq_batch = [pair[1] for pair in batch]


                    batch_preparer = E1BatchPreparer()
                    batch_e1 = batch_preparer.get_batch_kwargs(seq_batch, device=args.device)

                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        outputs = model(
                            input_ids=batch_e1["input_ids"],
                            within_seq_position_ids=batch_e1["within_seq_position_ids"],
                            global_position_ids=batch_e1["global_position_ids"],
                            sequence_ids=batch_e1["sequence_ids"],
                            past_key_values=None,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,
                        )

                    embeddings: torch.Tensor = outputs.embeddings  # (B, L, E)

                    # Boolean Selectors of shape (B, L) to get relevant tokens from logits/embeddings
                    # last_sequence_selector: True for tokens that are part of the last sequence (including boundary tokens) in case of multi-sequence input.
                    last_sequence_selector = batch_e1["sequence_ids"] == batch_e1["sequence_ids"].max(dim=1)[0][:, None]
                    # residue_selector: True for tokens that are part of the input sequence i.e not boundary tokens like 1, 2, <bos>, <eos>, <pad>, etc.
                    residue_selector = ~(batch_preparer.get_boundary_token_mask(batch_e1["input_ids"]))
                    # last_sequence_residue_selector: True for residues that are part of the last sequence (excluding boundary tokens)
                    last_sequence_residue_selector = last_sequence_selector & residue_selector


                    last_sequence_embeddings = [embeddings[i, last_sequence_residue_selector[i]].detach().to(torch.float32).cpu().numpy() for i in range(embeddings.shape[0])]


                # Save each sequence embedding
                for bi, (sid, s) in enumerate(batch):
                    li = len(s)
                    emb = last_sequence_embeddings[bi]  # [L, D]


                    safe_sid = re.sub(r"[^A-Za-z0-9._+-]", "_", sid)[:200]
                    out_path = msa_out_dir / f"{safe_sid}.npy"
                    np.save(out_path, emb)



    print("Done.")


if __name__ == "__main__":
    main()
