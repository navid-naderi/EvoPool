# EvoPool: Evolution-Guided Pooling of Protein Language Model Embeddings

This repository contains the implementation of EvoPool, an evolution-guided pooling method for protein language model (PLM) embeddings. The code supports:
- Generating training data from OpenProteinSet / UniClust30 MSAs (download + embedding extraction);
- Self-supervised primal-dual training of EvoPool (contrastive objective + uniformity constraint); and,
- Zero-shot evaluation on ProteinGym DMS substitutions using cosine similarity between pooled wildtype and variant embeddings.

## Abstract
Protein language models (PLMs) encode amino acid sequences into residue-level embeddings that must be pooled into fixed-size representations for downstream protein-level prediction tasks. Although these embeddings implicitly reflect evolutionary constraints, existing pooling strategies operate on single sequences and do not explicitly leverage information from homologous sequences or multiple sequence alignments. We introduce EvoPool, a self-supervised pooling framework that integrates evolutionary information from homologs directly into aggregated PLM representations using optimal transport. Our method constructs a fixed-size evolutionary anchor from an arbitrary number of homologous sequences and uses sliced Wasserstein distances to derive query protein embeddings that are geometrically informed by homologous sequence embeddings. Experiments across multiple state-of-the-art PLM families on the ProteinGym benchmark show that EvoPool consistently outperforms standard pooling baselines for variant effect prediction, demonstrating that explicit evolutionary guidance substantially enhances the functional utility of PLM representations.

![Overview of EvoPool.](https://github.com/navid-naderi/EvoPool/blob/main/assets/fig_teaser.png?raw=true)

## Setting Up the Environment
Create a new virtual Conda environment, called `evopool`, with the required libraries using the following commands:

```
conda create -n evopool python=3.12
conda activate evopool
pip install -r requirements.txt
```

## Training Data Generation
```
bash gen_data.sh
```

Running `gen_data.sh` does the following:
- Downloading UniClust30 MSA groups from OpenProteinSet containing A3M files
- Extracting residue-level embeddings for all sequences in each A3M

This script will create directories `./openproteinset_<num_MSAs>/` (containing downloaded A3M files under `uniclust30/<group>/a3m/...`) and `./openproteinset_<num_MSAs>_<plm>/` (containing extracted embeddings under a parallel tree). For each A3M file, embeddings are saved as `.../<msa_name>_embeds/<sequence_id>.npy`, where each `.npy` is an array of shape `[N, D]` (token embeddings). The variables `num_MSAs` (number of downloaded MSAs) and `plm` (PLM type) can be modified directly in `gen_data.sh`. 

## Self-Supervised Primal-Dual Training
```
python pretrain_evopool.py
```
This script is configured via the following variables:
- `plm_type`: one of `esm2_t33_650M_UR50D`, `esmc_600M`, `e1_600M`
- `num_ref_points`: number of SWE anchor points
- `tau_contrastive`: temperature for contrastive objective
- `eps_uniformity`: uniformity constraint threshold (e.g., -2)
- `lr` (primal learning rate), `dual_lr` (dual learning rate), `batch_size` (batch size), `num_epochs` (number of epochs), `random_seed` (random seed)

The script trains two torch neural network models:
- `swe_msa_length`: Low-level SWE pooling module used to embed homologs into a fixed-size representation and build the evolutionary anchor, and
- `swe_query`: High-level SWE pooling module used to pool query token embeddings given the evolutionary anchor.

The script also logs training curves to Weights & Biases and saves checkpoints to `./checkpoints/<wandb_project>/<run_id>/all_models_lambdas.pt`. The checkpoint includes: `swe_msa_length` weights, `swe_query` weights, and `MSA_lambdas` learned per-MSA Lagrangian dual multipliers.

## ProteinGym Evaluation (Zero-Shot DMS Substitutions)
```
python eval_evopool_proteingym.py
```
This script:
- loads a trained checkpoint from `./checkpoints/<wandb_project>/<run_id>/all_models_lambdas.pt`
- iterates over ProteinGym DMS substitution assays
- uses homolog sequences from ProteinGym-provided MSAs
- computes pooled embeddings and scores variants using cosine similarity
- writes per-assay results to `./results/dms/`

## Citation

If you make use of this repository, please cite our paper using the following BibTeX format:
```
@article{naderializadeh2026evopool,
  title={{EvoPool}: Evolution-Guided Pooling of Protein Language Model Embeddings},
  author={NaderiAlizadeh, Navid and Singh, Rohit},
  journal={bioRxiv},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
