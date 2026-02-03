import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import esm
import os
import gc
import itertools
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats
import wandb
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from swe import SWE_Pooling_AxisAligned
from Bio import SeqIO
import copy
from transformers import AutoModelForMaskedLM


# soft weights
def redundancy_weights_from_embeddings_batched(Z: torch.Tensor, beta: float = 50.0, eps: float = 1e-12):

    Zc = Z.detach()
    Zc = Zc / Zc.norm(dim=-1, keepdim=True).clamp_min(eps)
    S = Zc @ Zc.transpose(-1, -2)  # (B,M,M)

    B, M, _ = S.shape
    eye = torch.eye(M, device=S.device, dtype=torch.bool).unsqueeze(0)
    S = S.masked_fill(eye, -1e9)   # remove self

    rho = torch.exp(beta * S).sum(dim=-1)  # (B,M)
    rho = rho + 1.0                        # add self as baseline density

    w = 1.0 / rho
    w = w / w.sum(dim=1, keepdim=True)

    return w



def set_random_seed(seed):
    random.seed(seed)                # Python random module
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch CPU
    torch.cuda.manual_seed(seed)     # PyTorch CUDA (current device)
    torch.cuda.manual_seed_all(seed) # All CUDA devices, if using multi-GPU

    torch.backends.cudnn.deterministic = True  # Enforce determinism in cuDNN
    torch.backends.cudnn.benchmark = False     # Disables the inbuilt cudnn auto-tuner (for deterministic results)



def get_WT_seq(dms):

    NATURAL_AAS_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    first_mut_seq = dms.iloc[0]['mutated_sequence']
    first_mutation = dms.iloc[0]['mutant']

    wt_seq = copy.deepcopy(first_mut_seq)
    for mut in first_mutation.split(':'):

        wt_aa = mut[0]
        mu_aa = mut[-1]
        loc_aa = int(mut[1:-1]) - 1

        assert wt_seq[loc_aa] == mu_aa

        wt_seq = wt_seq[:loc_aa] + wt_aa + wt_seq[loc_aa + 1:]

    return wt_seq


PADDING_VALUE = 100


from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json, random
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

def safe_spearmanr(x, y, return_zero_for_constant=True):
    """
    Calculate Spearman correlation, optionally returning 0 for constant arrays.
    
    Args:
        return_zero_for_constant: If True, return (0, 1) when either array is constant.
                                  If False, return (nan, nan) like scipy default.
    """
    from collections import namedtuple

    # Create a SpearmanrResult-like namedtuple
    SpearmanrResult = namedtuple('SpearmanrResult', ['statistic', 'pvalue'])
    
    if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
        if return_zero_for_constant:
            return SpearmanrResult(statistic=0.0, pvalue=1.0)
        else:
            return SpearmanrResult(statistic=np.nan, pvalue=np.nan)
    
    return scipy.stats.spearmanr(x, y)


def analyze_embeddings_zeroshot(train_encodings, wt_embedding, train_labels):

    # 1) calculate the L2-distance between variant and WT embeddings and calculate the correlation
    emb_distances = np.linalg.norm(train_encodings - wt_embedding, axis=1)
    res_emb_distances = safe_spearmanr(emb_distances, train_labels)
    spearmanr_emb_distances = res_emb_distances.statistic

    # 2) calculate the L2-distance between NORMALIZED variant and WT embeddings (i.e., on the unit hypersphere) and calculate the correlation
    emb_distances_normalized = np.linalg.norm(normalize(train_encodings, axis=1) - normalize(wt_embedding, axis=1), axis=1)
    res_emb_distances_normalized = safe_spearmanr(emb_distances_normalized, train_labels)
    spearmanr_emb_distances_normalized = res_emb_distances_normalized.statistic

    # 3) calculate the cosine similarity between variant and WT embeddings and calculate the correlation
    emb_cosine_similarities = np.squeeze(cosine_similarity(train_encodings, wt_embedding), axis=-1)
    res_emb_cosine_similarities = safe_spearmanr(emb_cosine_similarities, train_labels)
    spearmanr_emb_cosine_similarities = res_emb_cosine_similarities.statistic


    return spearmanr_emb_distances, spearmanr_emb_distances_normalized, spearmanr_emb_cosine_similarities





import re

AA_RE = re.compile(r"[A-Z?]")

def clean_protein_seq(s: str) -> str:
    s = s.upper()                 # lowercase → uppercase
    return "".join(AA_RE.findall(s)) # this removes '.', '-', spaces, and also the literal '...'


def extract_embeddings(seq_list, plm_type, batch_size=8):

    if 'esm2' in plm_type:

        MAX_SEQ_LEN = 1022 # Define the limit

        data = [(i, clean_protein_seq(seq_list[i])[:MAX_SEQ_LEN]) for i in range(len(seq_list))] # change lower-case insertions to upper-case

        # Sort the data by sequence length; This groups similar-length sequences together, minimizing padding.
        data.sort(key=lambda x: len(x[1]), reverse=True)



        if plm_type == 'esm2_t6_8M_UR50D':
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            plm_emb_dim, repr_layers = 320, 6
        elif plm_type == 'esm2_t12_35M_UR50D':
            model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            plm_emb_dim, repr_layers = 480, 12
        elif plm_type == 'esm2_t30_150M_UR50D':
            model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            plm_emb_dim, repr_layers = 640, 30
        elif plm_type == 'esm2_t33_650M_UR50D':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            plm_emb_dim, repr_layers = 1280, 33
        elif plm_type == 'esm2_t36_3B_UR50D':
            model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            plm_emb_dim, repr_layers = 2560, 36
        else:
            raise Exception

        batch_converter = alphabet.get_batch_converter()
        model.cuda()
        model.eval()  # disables dropout for deterministic results

        # Extract per-residue representations
        # all_token_representations = []
        # Use a dictionary to store results by original index
        results_dict = {}
        with torch.no_grad():

            for b_ind in tqdm(range(int(np.ceil(len(data) / batch_size)))):
                seq_batch = data[b_ind * batch_size : (b_ind + 1) * batch_size]

                batch_labels, batch_strs, batch_tokens = batch_converter(seq_batch)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                results = model(batch_tokens.cuda(), repr_layers=[repr_layers], return_contacts=False)
                batch_token_representations = results["representations"][repr_layers].detach().cpu() # remove the eos and bos token embeddings
                
                # We want to slice from index 1 (after BOS)
                # up to the EOS token, which is at index (batch_lens[i] - 1)
                # Python slicing [1 : N] gives items 1, 2, ... N-1
                # So, [1 : batch_lens[i] - 1] gives reps from index 1 up to (batch_lens[i] - 2)
                
                for i, label in enumerate(batch_labels):
                    results_dict[label] = batch_token_representations[i, 1 : batch_lens[i] - 1]


                # Reconstruct the list in the original order
                del results
                del batch_token_representations
                del batch_tokens
                gc.collect()
                torch.cuda.empty_cache()

        # Reconstruct the list in the original order
        all_token_representations = [results_dict[i] for i in range(len(seq_list))]


    elif 'esmc' in plm_type: # https://huggingface.co/Synthyra/ESMplusplus_large


        data = [(i, clean_protein_seq(seq_list[i])) for i in range(len(seq_list))] # change lower-case insertions to upper-case

        if plm_type == 'esmc_600M':
            model = AutoModelForMaskedLM.from_pretrained('Synthyra/ESMplusplus_large', trust_remote_code=True)
        else:
            raise Exception

        device = 'cuda:0'
        model.eval()
        model = model.to(device)

        tokenizer = model.tokenizer

        # Extract per-residue representations
        all_token_representations = []
        with torch.no_grad():

            for b_ind in tqdm(range(int(np.ceil(len(data) / batch_size)))):
                seq_batch = [pair[1] for pair in data[b_ind * batch_size : (b_ind + 1) * batch_size]]
                
                tokenized = tokenizer(seq_batch, padding=True, return_tensors='pt').to(device)
                output = model(**tokenized) # get all hidden states with output_hidden_states=True
                batch_token_representations = output.last_hidden_state[:, 1:-1].detach().cpu() # remove the eos and bos token embeddings


                non_padded_embeddings = [batch_token_representations[i, 1:len(seq_batch[i])+1].detach().to(torch.float32).cpu() for i in range(len(seq_batch))]

                all_token_representations.extend(non_padded_embeddings)


    elif 'e1' in plm_type:

        from E1.batch_preparer import E1BatchPreparer
        from E1.modeling import E1ForMaskedLM

        device = 'cuda:0'

        if plm_type == 'e1_600M':
            model = E1ForMaskedLM.from_pretrained("Profluent-Bio/E1-600m")
        else:
            raise Exception

        model.eval()
        model = model.to(device)


        data = [(i, clean_protein_seq(seq_list[i])) for i in range(len(seq_list))] # change lower-case insertions to upper-case

        # Extract per-residue representations
        all_token_representations = []
        with torch.no_grad():

            for b_ind in tqdm(range(int(np.ceil(len(data) / batch_size)))):
                seq_batch = [pair[1] for pair in data[b_ind * batch_size : (b_ind + 1) * batch_size]]

                batch_preparer = E1BatchPreparer()
                batch_e1 = batch_preparer.get_batch_kwargs(seq_batch, device=device)

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

                last_sequence_embeddings = [embeddings[i, last_sequence_residue_selector[i]].detach().to(torch.float32).cpu() for i in range(embeddings.shape[0])]

                all_token_representations.extend(last_sequence_embeddings)

    return all_token_representations


def get_embeddings_swe_avg(dms_folder, assay, plm_type, msa_aggregated_lengthdepth, swe_query):

    dms = pd.read_csv(dms_folder + '/' + assay)

    # get the token-level PLM embeddings of all variant sequences

    CHUNK_SIZE = 2000  # Process the sequences in small chunks at a time
    embedding_chunks_parent_path = './embeddings_chunked/dms/{0}'.format(assay)

    all_chunk_paths = []
    for chunk_ind, chunk_start in enumerate(range(0, len(dms), CHUNK_SIZE)):
        chunk_path = '{0}/{1}_{2}_{3}.pt'.format(embedding_chunks_parent_path, plm_type, CHUNK_SIZE, chunk_ind)
        all_chunk_paths.append(chunk_path)
        if not os.path.exists(chunk_path): # generate the chunk embeddings
            print('Generating and saving embeddings for chunk {}'.format(chunk_ind))
            chunk_end = min(chunk_start + CHUNK_SIZE, len(dms))
            data_chunk = list(dms['mutated_sequence'][chunk_start:chunk_end])
            dms_label_chunk = list(dms['DMS_score'][chunk_start:chunk_end])
            dms_emb_chunk = extract_embeddings(data_chunk, plm_type)
            dms_emb_chunk = torch.stack(dms_emb_chunk, dim=0)
            os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
            torch.save([dms_emb_chunk, dms_label_chunk], chunk_path)

    print('Number of chunks:', len(all_chunk_paths))
    
    avg_scores_all, swe_scores_all, dms_labels = [], [], []


    # get WT embeddings
    wt_seq = get_WT_seq(dms)
    wt_emb = extract_embeddings([wt_seq], plm_type)[0].unsqueeze(0).to(device)
    avgpooled_wt_embedding = torch.mean(wt_emb[0], dim=0, keepdim=True).detach().cpu().numpy()
    msaswe_wt_embedding = swe_query(wt_emb, ref=msa_aggregated_lengthdepth).detach().cpu().numpy()

    seq_len = len(wt_seq)
    inference_batch_size = max(int(128 / ((seq_len * np.log(seq_len) / (100 * np.log(100))) ** 1)), 1) # adapt log-linearly to the length of the sequence, with B=64 at N=100batch_size, collate_fn=family_embeddings_supervised_collate_fn, shuffle=True)

    # get variant embeddings and labels in chunks
    for chunk_path in all_chunk_paths:#glob.glob(os.path.join(embedding_chunks_parent_path, "*.pt")):
        dms_emb_chunk, dms_label_chunk = torch.load(chunk_path)


        dms_labels.append(dms_label_chunk)

        avg_scores = get_zeroshot_scores(torch.mean(dms_emb_chunk, dim=1).detach().cpu().numpy(), avgpooled_wt_embedding)
        avg_scores_all.append(avg_scores)


        # pool all variants using the generated reference

        loader = DataLoader(dms_emb_chunk, batch_size=inference_batch_size, shuffle=False)

        for i, emb_batch in tqdm(
            enumerate(loader), total=len(loader)
        ):
            emb_batch = emb_batch.to(device)
            batch_size = len(emb_batch)
            # set the reference for each query sequence to the final aggregated MSAs
            swe_references = msa_aggregated_lengthdepth.repeat(batch_size, 1, 1) # [B, M, D]
            final_embeddings = swe_query(emb_batch, ref=swe_references) # [B, D]

            swe_scores = get_zeroshot_scores(final_embeddings.detach().cpu().numpy(), msaswe_wt_embedding)
            swe_scores_all.append(swe_scores)


        del dms_emb_chunk
        del final_embeddings
        del swe_references
        torch.cuda.empty_cache()


    dms_labels = np.concatenate(dms_labels, axis=0)
    avg_scores_all = np.concatenate(avg_scores_all, axis=1)
    swe_scores_all = np.concatenate(swe_scores_all, axis=1)

    del dms

    return avg_scores_all, swe_scores_all, dms_labels


def get_zeroshot_scores(variant_embs, wt_emb):
    
    # 1) calculate the L2-distance between variant and WT embeddings
    emb_distances = np.linalg.norm(variant_embs - wt_emb, axis=1)

    # 2) calculate the L2-distance between NORMALIZED variant and WT embeddings (i.e., on the unit hypersphere)
    emb_distances_normalized = np.linalg.norm(normalize(variant_embs, axis=1) - normalize(wt_emb, axis=1), axis=1)

    # 3) calculate the cosine similarity between variant and WT embeddings
    emb_cosine_similarities = np.squeeze(cosine_similarity(variant_embs, wt_emb), axis=-1)

    return np.stack((emb_distances, emb_distances_normalized, emb_cosine_similarities), axis=0)



def analyze_scores_zeroshot(model_scores, dms_labels):

    corr_list = []
    for ss in model_scores:
        corr_list.append(safe_spearmanr(ss, dms_labels).statistic)
    return tuple(corr_list)




if __name__ == '__main__':


    device = 'cuda'

    # main hyperparameters
    plm_type = 'esm2_t33_650M_UR50D' # one of ['e1_600M', 'esm2_t33_650M_UR50D', 'esmc_600M']
    num_ref_points = 128
    tau_contrastive = 0.1
    eps_uniformity = -2
    random_seed = 101
    lr = 5e-4
    dual_lr = 2e-1
    num_epochs = 40
    batch_size = 8
    
    msa_depth = 24 # number of homologs subsampled for inference

    set_random_seed(random_seed)

    if plm_type == 'esm2_t33_650M_UR50D':
        dim_plm = 1280
        embeds_root = "./openproteinset_5000_esm2_650M"
    elif plm_type == 'esmc_600M':
        dim_plm = 1152
        embeds_root = "./openproteinset_5000_esmc_600M"
    elif plm_type == 'e1_600M':
        dim_plm = 1280
        embeds_root = "./openproteinset_5000_e1_600M"
    else:
        raise Exception


    config = {'random_seed': random_seed,
    'lr': lr,
    'dual_lr': dual_lr,
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'plm_type': plm_type,
    'tau_contrastive': tau_contrastive,
    'num_ref_points': num_ref_points,
    'eps_uniformity': eps_uniformity,
    'msa_depth': msa_depth,
    }


    wandb_proj = 'EvoPool_UniClust30_OpenProteinSet_5000'
    run_id = f"{plm_type}_lr{lr}_duallr{dual_lr}_bs{batch_size}_tau{tau_contrastive}_ref{num_ref_points}_eps{eps_uniformity}_seed{random_seed}"

    # check if run is already done
    completed_flag = False
    api = wandb.Api()
    runs = api.runs(path='nnlab/{}'.format(wandb_proj))
    for run in runs:
        if run.name == run_id and run.state in ["finished"]:
            print('Run is completed and ready for eval')
            completed_flag = True
        else:
            pass

    if not completed_flag:
        print('Run not completed; terminating ...'.format())
        raise Exception


    model_save_path = './checkpoints/{0}/{1}/all_models_lambdas.pt'.format(wandb_proj, run_id)    
    
    swe_msa_length = SWE_Pooling_AxisAligned(d_in=dim_plm, num_ref_points=num_ref_points, learned_ref=True, flatten_embeddings=False)
    swe_query = SWE_Pooling_AxisAligned(d_in=dim_plm, num_ref_points=num_ref_points, learned_ref=False, flatten_embeddings=True)


    checkpoint = torch.load(model_save_path, weights_only=False)
    swe_msa_length.load_state_dict(checkpoint['swe_msa_length'])
    swe_query.load_state_dict(checkpoint['swe_query'])


    swe_msa_length.to(device)
    swe_query.to(device)

    swe_msa_length.eval()
    swe_query.eval()


    dms_folder = './DMS_ProteinGym_substitutions'
    dms_list = os.listdir(dms_folder)
    dms_list = [f for f in dms_list if f.endswith('.csv')]
    

    print("dms_list", len(dms_list), dms_list)

    msa_folder = './DMS_msa_files' # these are just alignments provided by proteingym


    for assay in dms_list:


        print('*'*50)
        print("assay", assay)

        results_folder = './results/dms/{0}/{1}'.format(assay[:-4], wandb_proj)
        os.makedirs(results_folder, exist_ok=True)


        results_file = results_folder + '/{0}_InferenceMSADepth{1}.pt'.format(run_id, msa_depth)

        print("results_file", results_file)

        if os.path.exists(results_file):
            print('results already exist! skipping ...')
            continue


        # find MSA

        df_meta = pd.read_csv('./DMS_substitutions.csv')

        # Create a dictionary mapping DMS_filename to MSA_filename
        dms_to_msa = dict(zip(df_meta['DMS_filename'], df_meta['MSA_filename']))
        msa_file_name = dms_to_msa[assay]

        msa_file = '{0}/{1}'.format(msa_folder, msa_file_name) # original proteingym-provided msa

        msa = [clean_protein_seq(str(record.seq)) for record in SeqIO.parse(msa_file, "fasta-blast")]

        token_representations_msa_subsampled = extract_embeddings(msa[1:msa_depth+1], plm_type) # sampling the first few aligned sequences; skip the first sequence, since it's the query sequence

        msa_padded = nn.utils.rnn.pad_sequence(token_representations_msa_subsampled, batch_first=True, padding_value=PADDING_VALUE, padding_side='right').to(device)
        msa_lens = [len(msa_embs) for msa_embs in token_representations_msa_subsampled]
        msa_mask = torch.zeros((msa_depth, np.max(msa_lens)), dtype=torch.bool, device=device)
        for i in range(msa_depth):
            msa_mask[i, :msa_lens[i]] = True

        # generate anchors using the sub-sampled MSAs

        # stack the MSA sequences and aggregate them across the length dimension using the first SWE

        msa_aggregated_lengthdim = swe_msa_length(msa_padded, mask=msa_mask) # [msa_depth x M x D]

        print("msa_aggregated_lengthdim", msa_aggregated_lengthdim.shape)

        _, num_ref_points, D = msa_aggregated_lengthdim.shape
       
        msa_aggregated_lengthdim_flattened = msa_aggregated_lengthdim.view(-1, num_ref_points * D).unsqueeze(0) # [1, msa_depth, M*D]
        msa_weights = redundancy_weights_from_embeddings_batched(msa_aggregated_lengthdim_flattened) # [1, msa_depth]


        # weighted averaging
        msa_aggregated_lengthdepth = (msa_weights.unsqueeze(-1).unsqueeze(-1) * msa_aggregated_lengthdim.unsqueeze(0)).sum(dim=1) # [1, M, D]

        print("msa_aggregated_lengthdepth", msa_aggregated_lengthdepth.shape)

        avg_scores_all, swe_scores_all, dms_labels = get_embeddings_swe_avg(dms_folder, assay, plm_type, msa_aggregated_lengthdepth, swe_query)


        # get results of AVG pooling
        results_plm_avgpool = analyze_scores_zeroshot(avg_scores_all, dms_labels)
        print("results_plm_avgpool", results_plm_avgpool)

        # get results for EvoPool
        results_evopool = analyze_scores_zeroshot(swe_scores_all, dms_labels)
        print("results_evopool", results_evopool)


        torch.save([results_plm_avgpool, results_evopool], results_file)

        del msa_aggregated_lengthdim
        del msa_aggregated_lengthdepth
        del token_representations_msa_subsampled
        torch.cuda.empty_cache()