#!/usr/bin/env python3
"""
download_exact_msas_with_a3m.py

Download EXACTLY N OpenProteinSet groups (under a given prefix, e.g. "uniclust30/")
that actually contain A3M files. You can choose to download only the canonical
A3M file ("a3m/uniclust30.a3m") or ALL files under "a3m/".
"""

import argparse
from pathlib import Path
from typing import Dict, List, Iterable
import random

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BUCKET = "openfold"


# ------------------------------- S3 helpers -------------------------------- #
def s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))

def list_objects(prefix: str) -> Iterable[dict]:
    """Yield all S3 objects under prefix (non-directory keys)."""
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            if not obj["Key"].endswith("/"):
                yield obj


# ---------------------------- Grouping & filtering ------------------------- #
def infer_groups(objs: Iterable[dict], top_prefix: str) -> Dict[str, List[str]]:
    """
    Group object keys by their first-level folder after top_prefix.
    For example:
        Key: 'uniclust30/ABC123/a3m/uniclust30.a3m'
        Group: 'uniclust30/ABC123/'
    Returns: { group_prefix: [keys...] }
    """
    groups: Dict[str, List[str]] = {}
    top = top_prefix if top_prefix.endswith("/") else top_prefix + "/"
    for o in objs:
        key = o["Key"]
        if not key.startswith(top):
            continue
        rest = key[len(top):]
        parts = rest.split("/", 1)
        if len(parts) < 2:
            # skip objects directly at top (unexpected)
            continue
        group_name = parts[0]
        group_prefix = f"{top}{group_name}/"
        groups.setdefault(group_prefix, []).append(key)
    return groups

def groups_with_a3m(groups: Dict[str, List[str]]) -> List[str]:
    """Return group prefixes that contain at least one key under '/a3m/'."""
    good = []
    for gp, keys in groups.items():
        if any("/a3m/" in k for k in keys):
            good.append(gp)
    return sorted(good)


# --------------------------------- Download -------------------------------- #
def filter_keys_for_download(keys: List[str], mode: str) -> List[str]:
    """
    mode:
      - 'canonical': only '.../a3m/uniclust30.a3m' if present; otherwise take the first A3M file found
      - 'all_a3m' : all files under '.../a3m/'
    """
    a3m_keys = [k for k in keys if "/a3m/" in k]
    if mode == "all_a3m":
        return a3m_keys
    # canonical
    preferred = [k for k in a3m_keys if k.endswith("/a3m/uniclust30.a3m")]
    if preferred:
        return preferred
    return a3m_keys[:1]  # fallback within the group to a single A3M file

def download_keys(keys: List[str], out_root: Path, overwrite: bool = False):
    client = s3_client()
    iterable = tqdm(keys, desc="Downloading", unit="file") if tqdm else keys
    for key in iterable:
        target = out_root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            continue
        try:
            client.download_file(BUCKET, key, str(target))
        except ClientError as e:
            print(f"[WARN] failed {key}: {e}")


# ---------------------------------- Main ----------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="Top-level prefix, e.g. 'uniclust30/'")
    ap.add_argument("--num-groups", type=int, required=True,
                    help="EXACT number of groups to download that contain A3M files.")
    ap.add_argument("--out", required=True, help="Local output directory.")
    ap.add_argument("--random", action="store_true",
                    help="Randomly sample the groups (default: take the first N after sorting).")
    ap.add_argument("--download-mode", choices=["canonical", "all_a3m"], default="canonical",
                    help="What to download per group: only canonical A3M (default) or all files under a3m/.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Enumerating s3://{BUCKET}/{args.prefix} ...")
    objs = list(list_objects(args.prefix))
    if not objs:
        raise SystemExit("No objects found; check --prefix.")

    print("Building groups ...")
    groups = infer_groups(objs, args.prefix)
    print(f"Total groups found under '{args.prefix}': {len(groups)}")

    print("Filtering to groups that contain A3M ...")
    a3m_groups = groups_with_a3m(groups)
    print(f"Groups with A3M: {len(a3m_groups)}")

    if len(a3m_groups) < args.num_groups:
        raise SystemExit(f"Requested {args.num_groups} groups, but only {len(a3m_groups)} groups contain A3M.")

    # Pick EXACTLY N groups from those that have A3Ms
    if args.random:
        random.shuffle(a3m_groups)
    chosen = a3m_groups[: args.num_groups]

    # Decide which keys to download
    to_download: List[str] = []
    groups_no_files = 0
    for gp in chosen:
        keys = groups[gp]
        sel = filter_keys_for_download(keys, args.download_mode)
        if not sel:
            groups_no_files += 1
            continue
        to_download.extend(sel)

    # Sanity: we guaranteed groups have A3M, but in theory sel could be empty only if keys vanished mid-run
    if groups_no_files:
        print(f"[WARN] {groups_no_files} chosen groups produced no selectable files (race or ACL change?).")

    print(f"Selected groups: {len(chosen)} (exact)")
    print(f"Files to download (mode='{args.download_mode}'): {len(to_download)}")

    download_keys(to_download, out_root, overwrite=args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
