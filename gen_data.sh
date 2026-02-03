#!/bin/bash

num_MSAs=5000
plm='esmc_600M' # one of ('esm2_650M' 'esmc_600M' 'e1_600M')

# Step 1: Download MSAs (comment if already downloaded)
python download_exact_msas_with_a3m.py --prefix uniclust30/ --num-groups $num_MSAs --random --out ./openproteinset_${num_MSAs}

# Step 2: Generate embeddings

echo "Running with PLM: $plm"

python extract_token_embeddings_from_msas.py --msa-root ./openproteinset_${num_MSAs}/uniclust30 --out-root ./openproteinset_${num_MSAs}_${plm} --batch-size 64 --plm $plm