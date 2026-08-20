"""Download UCLA CNP rest fMRIPrep MNI images for CONTROL and SCHZ only.

Files go to D:\\Leakage_Audit_data (C: is full). About 172 x 95 MB ≈ 16 GB.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/Leakage_Audit_data")
INDEX = DATA / "nilearn_data/ds000030/ds000030_R1.0.4/uncompressed/urls.json"
OUT = DATA / "ds000030"
PHENO = DATA / "participants.tsv"
BASE = "https://s3.amazonaws.com/openneuro/ds000030/ds000030_R1.0.4/uncompressed/"


def sid_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("_")[0]


def download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            expected = int(r.headers.get("Content-Length", "0") or 0)
    except Exception:
        expected = 0
    if dest.exists() and expected and dest.stat().st_size == expected:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            if expected and tmp.stat().st_size != expected:
                raise IOError(f"size {tmp.stat().st_size} != {expected}")
            tmp.replace(dest)
            return
        except Exception as exc:
            last_err = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed {url}: {last_err}")


def main():
    urls = json.loads(INDEX.read_text(encoding="utf-8"))
    ph = pd.read_csv(PHENO, sep="\t")
    keep = ph[(ph["diagnosis"].isin(["CONTROL", "SCHZ"])) & (ph["rest"] == 1.0)]
    ids = set(keep["participant_id"].astype(str))
    wanted = [
        u for u in urls
        if sid_from_url(u) in ids
        and (
            "task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz" in u
            or "task-rest_bold_confounds.tsv" in u
        )
    ]
    wanted = sorted(set(wanted))
    print(f"subjects {len(ids)}  files {len(wanted)}", flush=True)
    t0 = time.time()
    for i, url in enumerate(wanted, 1):
        rel = url.split("/uncompressed/", 1)[1]
        dest = OUT / rel
        download(url, dest)
        if i % 10 == 0 or i == len(wanted):
            print(f"{i}/{len(wanted)}  {rel}  ({time.time()-t0:.0f}s)", flush=True)
    print("done", OUT)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR", exc, file=sys.stderr)
        raise
