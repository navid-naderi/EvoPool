import torch
assert torch.cuda.is_available(), "CUDA is not available (no GPU / driver not visible)."

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import esm
import os
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

import gzip
import re
from pathlib import Path

_SAN_RE = re.compile(r"[^A-Za-z0-9._+-]")

def sanitize_sid(s: str, maxlen: int = 200) -> str:
    return _SAN_RE.sub("_", s)[:maxlen]

def read_first_a3m_id(a3m_path: Path) -> str | None:
    # reads only until the first header line
    opener = gzip.open if str(a3m_path).endswith(".gz") else open
    with opener(a3m_path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                return line[1:].split()[0]
    return None

def find_matching_a3m_file(a3m_dir: Path, embeds_dir: Path) -> Path | None:
    # embeds_dir.name is typically "uniclust30_embeds" => base "uniclust30"
    name = embeds_dir.name
    base = name[:-7] if name.endswith("_embeds") else name  # remove "_embeds"
    # try common cases
    cands = [
        a3m_dir / f"{base}.a3m",
        a3m_dir / f"{base}.a3m.gz",
    ]
    for p in cands:
        if p.exists():
            return p
    # fallback: any a3m-like file
    for pat in ("*.a3m", "*.a3m.gz"):
        hit = next(a3m_dir.glob(pat), None)
        if hit is not None:
            return hit
    return None



def set_random_seed(seed):
    random.seed(seed)                # Python random module
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch CPU
    torch.cuda.manual_seed(seed)     # PyTorch CUDA (current device)
    torch.cuda.manual_seed_all(seed) # All CUDA devices, if using multi-GPU

    torch.backends.cudnn.deterministic = True  # Enforce determinism in cuDNN
    torch.backends.cudnn.benchmark = False     # Disables the inbuilt cudnn auto-tuner (for deterministic results)


PADDING_VALUE = 100


from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json, random
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
import io, time

import torch


def redundancy_weights_from_embeddings_batched(
    Z: torch.Tensor,
    is_valid_msa: torch.Tensor,
    beta: float = 50.0,
    eps: float = 1e-12,
):
    """
    Z: (B, M, D)
    is_valid_msa: (B, M) binary {0,1}
    returns w: (B, M), sum to 1 over valid MSAs, zero on invalid ones
    """

    # Detach to save memory / avoid gradients
    Zc = Z.detach()

    # Normalize embeddings
    Zc = Zc / Zc.norm(dim=-1, keepdim=True).clamp_min(eps)

    # Cosine similarity
    S = Zc @ Zc.transpose(-1, -2)  # (B, M, M)

    B, M, _ = S.shape
    device = S.device

    # Masks
    eye = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0)
    valid_i = is_valid_msa.bool().unsqueeze(2)  # (B,M,1)
    valid_j = is_valid_msa.bool().unsqueeze(1)  # (B,1,M)

    # Remove self-similarity
    S = S.masked_fill(eye, -1e9)

    # Remove invalid MSAs from density computation
    S = S.masked_fill(~(valid_i & valid_j), -1e9)

    # Soft density
    rho = torch.exp(beta * S).sum(dim=-1)  # (B,M)

    # Add self baseline ONLY for valid MSAs
    rho = rho + is_valid_msa.float()

    # Inverse-density weights
    w = torch.zeros_like(rho)
    w_valid = is_valid_msa.bool()

    w[w_valid] = 1.0 / rho[w_valid].clamp_min(eps)

    # Normalize over valid MSAs only
    w_sum = w.sum(dim=1, keepdim=True).clamp_min(eps)
    w = w / w_sum

    return w




class OpenProteinSetQueryMSAContrastiveConcat(Dataset):
    def __init__(
        self,
        embeds_root: str | Path,
        M: int = 8,  # you can ignore this after changes, or set max_depth=M
        dtype: str = "float32",
        embeds_subdir: str = "uniclust30_embeds",
        seed: Optional[int] = None,
        min_depth: int = 8,
        max_depth: int = 32,
        # same_depth_for_both_views: bool = True,
    ):
        self.embeds_root = Path(embeds_root).resolve()
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.embeds_subdir = embeds_subdir
        self.rng = random.Random(seed)

        self.min_depth = int(min_depth)
        self.max_depth = int(max_depth)
        if self.min_depth < 0 or self.max_depth < 0 or self.min_depth > self.max_depth:
            raise ValueError(f"Bad depth range: min_depth={min_depth}, max_depth={max_depth}")

        index: List[Tuple[Path, Path, List[Path], Path]] = []  # (msa_dir, embeds_dir, all_seq_files, query_file)

        print("Iterating over the MSA directories to build the dataset ...")


        embeds_dir_list = [
            grp / "a3m" / self.embeds_subdir
            for grp in self.embeds_root.iterdir()
            if (grp / "a3m" / self.embeds_subdir).is_dir()
        ]



        for embeds_dir in tqdm(embeds_dir_list):
            if not embeds_dir.is_dir():
                continue

            seq_files = sorted(embeds_dir.glob("*.npy"))
            if not seq_files:
                continue

            a3m_dir = embeds_dir.parent              # .../<msa>/a3m
            msa_dir = a3m_dir.parent                 # .../<msa>

            a3m_file = find_matching_a3m_file(a3m_dir, embeds_dir)
            qfile = None

            if a3m_file is not None:
                qid = read_first_a3m_id(a3m_file)
                if qid is not None:
                    qsafe = sanitize_sid(qid)
                    cand = embeds_dir / f"{qsafe}.npy"
                    if cand.exists():
                        qfile = cand

            if qfile is None:
                # last-resort fallback if the query id file wasn't found:
                # just pick one deterministically (e.g., first .npy).
                # (If you want "longest", we can do that without loading full arrays too.)
                qfile = seq_files[0]

            index.append((msa_dir, embeds_dir, seq_files, qfile))

        if not index:
            raise RuntimeError("No valid MSAs found (empty dirs?).")

        self.index = index



        # Labels per MSA directory
        uniq, seen = [], set()
        for msa_dir, *_ in self.index:
            if msa_dir not in seen:
                uniq.append(msa_dir); seen.add(msa_dir)
        self.dir_to_label: Dict[Path, int] = {d: i for i, d in enumerate(uniq)}
        self.label_names: List[str] = [d.name for d in uniq]

    def __len__(self) -> int:
        return len(self.index)



    def _load_tensor(self, p: Path) -> torch.Tensor:
        arr = np.load(p)
        if arr.dtype == np.float16 and self.dtype == np.float32:
            arr = arr.astype(np.float32, copy=False)
        elif arr.dtype == np.float32 and self.dtype == np.float16:
            arr = arr.astype(np.float16, copy=False)
        return torch.from_numpy(arr)
    


    def _get_rng(self):
        wi = get_worker_info()
        if wi is None:
            return self.rng  # main process

        # one RNG per worker process
        if not hasattr(self, "_worker_rng"):
            # wi.seed is already a big int unique per worker; mix in worker id
            seed = int(wi.seed) + 1000003 * int(wi.id)
            self._worker_rng = random.Random(seed)
        return self._worker_rng



    def _sample_k(self, pool: List[Path], k: int) -> List[Path]:
        rng = self._get_rng()
        k = int(k)
        if k <= 0:
            return []
        if not pool:
            return []

        # without replacement if possible, otherwise with replacement
        if len(pool) >= k:
            return rng.sample(pool, k)
        return [rng.choice(pool) for _ in range(k)]


    def __getitem__(self, i: int):
        msa_dir, embeds_dir, seq_files, qfile = self.index[i]
        label = self.dir_to_label[msa_dir]

        q = self._load_tensor(qfile)  # [Lq, D]

        pool = [p for p in seq_files if p != qfile]
        rng = self._get_rng()

        k1 = rng.randint(self.min_depth, self.max_depth)
        k2 = rng.randint(self.min_depth, self.max_depth)

        picks1 = self._sample_k(pool, k1)
        picks2 = self._sample_k(pool, k2)

        view1 = [self._load_tensor(p) for p in picks1]
        view2 = [self._load_tensor(p) for p in picks2]

        meta = {
            "msa_dir": str(msa_dir),
            "embeds_dir": str(embeds_dir),
            "query_file": str(qfile),
            "picked1": [str(p) for p in picks1],
            "picked2": [str(p) for p in picks2],
            "depth1": len(picks1),
            "depth2": len(picks2),
            "label_name": self.label_names[label],
        }
        return q, view1, view2, label, meta

def pad_collate_contrastive_concat(batch, pad_value: float = 0.0):
    queries, views1, views2, labels, metas = zip(*batch)
    B = len(batch)
    D = queries[0].shape[1]

    # pad queries
    Lq_max = max(q.shape[0] for q in queries)
    Q = queries[0].new_full((B, Lq_max, D), pad_value)
    Q_mask = torch.zeros((B, Lq_max), dtype=torch.bool, device=queries[0].device)
    for i, q in enumerate(queries):
        L = q.shape[0]
        Q[i, :L, :] = q
        Q_mask[i, :L] = True

    # variable depths per sample
    M1_max = max(len(v) for v in views1) if B > 0 else 0
    M2_max = max(len(v) for v in views2) if B > 0 else 0
    M_max = max(M1_max, M2_max)

    # compute Lm_max across all picked tensors
    Lm_max = 0
    for lst in list(views1) + list(views2):
        for x in lst:
            if x.shape[0] > Lm_max:
                Lm_max = x.shape[0]

    if M_max == 0:
        MSA = queries[0].new_full((B, 0, 0, D), pad_value)
        MSA_mask = torch.zeros((B, 0, 0), dtype=torch.bool, device=queries[0].device)
    else:
        # pad sequence-count dim to M_max for each view, so total is 2*M_max
        MSA = queries[0].new_full((B, 2 * M_max, Lm_max, D), pad_value)
        MSA_mask = torch.zeros((B, 2 * M_max, Lm_max), dtype=torch.bool, device=queries[0].device)

        for i in range(B):
            # view1 in [0, M_max)
            for j, x in enumerate(views1[i]):
                L = x.shape[0]
                if j >= M_max:
                    break
                MSA[i, j, :L, :] = x
                MSA_mask[i, j, :L] = True

            # view2 in [M_max, 2*M_max)
            for j, x in enumerate(views2[i]):
                L = x.shape[0]
                if j >= M_max:
                    break
                MSA[i, M_max + j, :L, :] = x
                MSA_mask[i, M_max + j, :L] = True

    y = torch.tensor(labels, dtype=torch.long)
    return Q, Q_mask, MSA, MSA_mask, y, list(metas)







def contrastive_loss(z, tau):

    # z is a 2B x D matrix, where the first B rows include the first B embeddings and the second B rows include the second B embeddings (augmentations)
    

    N = z.shape[0]
    batch_size = N // 2


    sim = torch.nn.CosineSimilarity(dim=2)(z.unsqueeze(1), z.unsqueeze(0)) / tau

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    # create a mask to separate negative samples from positive ones
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    negative_samples = sim[mask].reshape(N, -1)
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    
    labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()

    loss = nn.CrossEntropyLoss()(logits, labels)

    return loss






def lunif_batched(x: torch.Tensor, mask: torch.Tensor, t: float = 2.0, eps: float = 1e-12):
    """
    x:    (B, N, D) embeddings
    mask: (B, N) binary/bool mask. Entries with 0 are discarded.
    Returns: (B,) Lunif value per batch item, computed over valid pairs only.
    """
    assert x.dim() == 3, f"x must be (B,N,D), got {tuple(x.shape)}"
    assert mask.dim() == 2, f"mask must be (B,N), got {tuple(mask.shape)}"
    B, N, D = x.shape
    assert mask.shape == (B, N), f"mask must be shape (B,N) = {(B,N)}, got {tuple(mask.shape)}"

    m = mask.to(dtype=torch.bool, device=x.device)

    # normalize to unit hypersphere
    x_norm = nn.functional.normalize(x, p=2, dim=-1, eps=eps)  # (B,N,D)

    # pairwise squared distances
    d2 = torch.cdist(x_norm, x_norm, p=2).pow(2)  # (B,N,N)

    # upper triangle (i < j)
    tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1)  # (N,N)

    # valid pairs: i<j and both valid in mask
    pair_valid = tri.unsqueeze(0) & (m.unsqueeze(2) & m.unsqueeze(1))  # (B,N,N)

    # compute log-mean-exp of (-t * d2) over valid pairs per batch item
    a = (-t * d2).masked_fill(~pair_valid, float("-inf"))  # (B,N,N)

    # count valid pairs per batch item (as tensor for broadcasting)
    K = pair_valid.sum(dim=(1, 2)).to(dtype=x.dtype)  # (B,)

    # if K==0 (fewer than 2 valid points), define output as 0 (or change to -inf if preferred)
    lse = torch.logsumexp(a.view(B, -1), dim=1)  # (B,)
    out = lse - torch.log(K.clamp_min(1.0))

    out = torch.where(K > 0, out, torch.zeros_like(out))
    return out



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
    # 'msa_depth': msa_depth,
    }

    wandb_proj = 'EvoPool_UniClust30_OpenProteinSet_5000'
    run_id = f"{plm_type}_lr{lr}_duallr{dual_lr}_bs{batch_size}_tau{tau_contrastive}_ref{num_ref_points}_eps{eps_uniformity}_seed{random_seed}"

    # check if run is already done

    api = wandb.Api()
    try:
        runs = api.runs(path='nnlab/{}'.format(wandb_proj))
        _ = len(runs)
    except (ValueError, Exception) as e:
        # Project does not exist or other error; skip
        print(f"Project not found or error occurred: {e}")
        runs = []

    for run in runs:
        if run.name == run_id and run.state in ["finished", "running"]:
            print('Run already completed or active! Terminating ...')
            raise Exception
        else:
            pass


    model_save_path = './checkpoints/{0}/{1}/all_models_lambdas.pt'.format(wandb_proj, run_id)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    wandb.init(
            project=wandb_proj,
            name=run_id,
            config=config,
            )


    ds = OpenProteinSetQueryMSAContrastiveConcat(
    embeds_root=embeds_root,
    dtype="float32",
    embeds_subdir="uniclust30_embeds",
    seed=random_seed,
    min_depth=8,
    max_depth=24,
    )

    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_contrastive_concat, num_workers=4)

    swe_msa_length = SWE_Pooling_AxisAligned(d_in=dim_plm, num_ref_points=num_ref_points, learned_ref=True, flatten_embeddings=False)
    swe_query = SWE_Pooling_AxisAligned(d_in=dim_plm, num_ref_points=num_ref_points, learned_ref=False, flatten_embeddings=True)

    swe_msa_length.to(device)
    swe_query.to(device)

    optimizer = torch.optim.AdamW(list(swe_msa_length.parameters()) + list(swe_query.parameters()), lr=lr)

    num_MSAs = len(ds.dir_to_label)
    MSA_lambdas = torch.zeros(num_MSAs, requires_grad=False).to(device)

    swe_msa_length.train()
    swe_query.train()

    print('Beginning Training ...')
    for epoch in range(num_epochs):

        print('Epoch {}'.format(epoch))

        for batch_ind, batch in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):


            q, q_mask, msa, msa_mask, y, _ = batch

            q = q.to(device) # [B, Lq_max, D]
            q_mask = q_mask.to(device) # [B, Lq_max]
            msa = msa.to(device) # [B, 2*M_max, Lm_max, D]   (M_max may be < M if some MSAs have few sequences)
            msa_mask = msa_mask.to(device) # [B, 2*M_max, Lm_max]
            y = y.to(device)

            B, M_max, Lm_max, D = msa.shape

            M_max = int(M_max / 2)


            msa = msa.view(-1, Lm_max, D) # [2*B*M_max, Lm_max, D]


            msa_mask_flat = msa_mask.view(-1, Lm_max) # [2*B*M_max, Lm_max]


            # process in minibatches
            swe_training_minibatch_size = 128
            msa_aggregated_lengthdim = []
            for swe_b_ind in range(int(np.ceil(len(msa) / swe_training_minibatch_size))):
                msa_minibatch = msa[swe_b_ind * swe_training_minibatch_size : (swe_b_ind + 1) * swe_training_minibatch_size]
                msa_mask_minibatch = msa_mask_flat[swe_b_ind * swe_training_minibatch_size : (swe_b_ind + 1) * swe_training_minibatch_size]
                msa_aggregated_lengthdim.append(swe_msa_length(msa_minibatch, mask=msa_mask_minibatch))
            msa_aggregated_lengthdim = torch.cat(msa_aggregated_lengthdim, dim=0)


            # separate each length-aggregated MSA embedding and aggregate them across the depth dimension using the second SWE
            msa_aggregated_lengthdim = msa_aggregated_lengthdim.view(B, -1, num_ref_points, D) # [B, 2*M_max, M, D]
            msa_aggregated_lengthdim = torch.cat([msa_aggregated_lengthdim[:, :M_max], msa_aggregated_lengthdim[:, M_max:]], dim=0) # [2*B, M_max, M, D] (first half of rows corresponds to first view)


            is_valid_msa = 1.0 * (msa_mask.sum(-1) > 0) # [B, 2*M_max]
            is_valid_msa_separated_views = torch.cat([is_valid_msa[:, :M_max], is_valid_msa[:, M_max:]], dim=0) # [2*B, M_max] (first half of rows corresponds to first view)

            msa_aggregated_lengthdim_flattened = msa_aggregated_lengthdim.view(2 * B, -1, num_ref_points * D) # [2*B, M_max, M*D]
            msa_weights = redundancy_weights_from_embeddings_batched(msa_aggregated_lengthdim_flattened, is_valid_msa_separated_views) # [2*B, M_max]


            # weighted averaging
            msa_aggregated_lengthdepth = (msa_weights.unsqueeze(-1).unsqueeze(-1) * msa_aggregated_lengthdim).sum(dim=1) # [2*B, M, D]


            # set the reference for each query sequence to the final aggregated MSAs
            query_swe_references = msa_aggregated_lengthdepth # [2*B, M, D]


            # aggregate the query using the MSA-based references (each query has two subsampled MSAs, hence two references)
            query_repeated = q.repeat(2, 1, 1) # [2*B, Lq_max, D]
            query_mask_repeated = q_mask.repeat(2, 1) # [2*B, Lq_max, D]
            query_embeddings = swe_query(query_repeated, mask=query_mask_repeated, ref=query_swe_references) # [2*B, D]


            # objective = contrastive loss on query embeddings (2 different subsampled MSAs)
            obj = contrastive_loss(query_embeddings, tau_contrastive)


            # constraints based on uniformity losses of each MSA

            # sequences within each subsampled MSA is viewed as query sequences and pooled using references generated from the other subsampled MSA
            query_swe_references_separated_views = torch.transpose(query_swe_references.view(2, B, -1, D), 0, 1) # [B, 2, M, D]
            query_swe_references_separated_views_swapped = torch.stack([query_swe_references_separated_views[:, 1], query_swe_references_separated_views[:, 0]], dim=1) # [B, 2, M, D]
            query_swe_references_repeated_swapped_views = query_swe_references_separated_views_swapped.unsqueeze(2).expand(B, 2, M_max, num_ref_points, D).reshape(-1, num_ref_points, D) # [2*B*M_max, M, D]



            # msa: [2*B*M_max, Lm_max, D]  (i-th group of 2*M_max rows belong to the first/second views of the i-th MSA, i=0,...,B-1)
            # msa_mask_flat: [2*B*M_max, Lm_max]  (i-th group of 2*M_max rows belong to the first/second views of the i-th MSA, i=0,...,B-1)
            # query_swe_references_swapped_views: [2*B*M_max, M, D] (i-th group of 2*M_max rows belong to the second/first views of the i-th MSA, i=0,...,B-1)

            # process in minibatches
            msa_embeddings_swapped_views = []
            for swe_b_ind in range(int(np.ceil(len(msa) / swe_training_minibatch_size))):
                msa_minibatch = msa[swe_b_ind * swe_training_minibatch_size : (swe_b_ind + 1) * swe_training_minibatch_size]
                msa_mask_minibatch = msa_mask_flat[swe_b_ind * swe_training_minibatch_size : (swe_b_ind + 1) * swe_training_minibatch_size]
                ref_minibatch = query_swe_references_repeated_swapped_views[swe_b_ind * swe_training_minibatch_size : (swe_b_ind + 1) * swe_training_minibatch_size]
                msa_embeddings_swapped_views.append(swe_query(msa_minibatch, mask=msa_mask_minibatch, ref=ref_minibatch))
            msa_embeddings_swapped_views = torch.cat(msa_embeddings_swapped_views, dim=0)

            # regroup the embeddings belonging to the same MSA together
            msa_embeddings_swapped_views_grouped = msa_embeddings_swapped_views.view(B, -1, D) # [B, 2M_max, D]
            is_valid_msa_grouped = 1.0 * (msa_mask_flat.sum(-1) > 0).view(B, -1) # [B, 2*M_max]


            unif_loss = lunif_batched(msa_embeddings_swapped_views_grouped, is_valid_msa_grouped)

            batch_lambdas = MSA_lambdas[y]

            constraint_violations = (unif_loss - eps_uniformity)

            constraint_violation_term = torch.sum(batch_lambdas * constraint_violations)
    
            lagrangian = obj + constraint_violation_term

                
            optimizer.zero_grad()
            lagrangian.backward()


            # clip gradients
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm=1.0)

            optimizer.step()

            MSA_lambdas[y] = batch_lambdas + dual_lr * constraint_violations.detach()
            MSA_lambdas.data.clamp_(0)


            results = {
                    "train/step": (epoch * len(train_loader) * batch_size) + (batch_ind * batch_size),
                    "train/loss": obj.item(),
                    "train/MSA_lambdas": MSA_lambdas.mean().item(),
                    "train/uniformity_loss": torch.Tensor(unif_loss).mean().item(),
                    "train/epoch": epoch,
                }
            wandb.log(results)

        torch.save({
            'swe_msa_length': swe_msa_length.state_dict(),
            'swe_query': swe_query.state_dict(),
            'MSA_lambdas': MSA_lambdas.detach().cpu().numpy()
        }, model_save_path)

    print('Finished Training ...')