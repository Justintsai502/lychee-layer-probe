#!/usr/bin/env python3
"""
Recompute excess shares and acoustic_share in result JSONs from their stored eta values.

The eta values (share of variance explained by speaker / by sentence) are exact, and on
a balanced table the chance floor depends only on them and the table size, so the
corrected numbers can be derived without re-running the model:

  python lyprobe/rescore.py                 # every results/*.json, in place

Hard deps: none beyond the repo
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import RESULTS_DIR  # noqa: E402
from lyprobe.metrics import shares_from_eta  # noqa: E402

METHOD = "omega2: chance floor from residual variance (k - 1) * ms_resid"


def rescore(res):
    n, ks, kt = res["n_clips"], res["n_speakers"], res["n_sentences"]
    for row in res["points"] + res["baselines"]:
        new = shares_from_eta(row["eta_spk"], row["eta_sent"], n, ks, kt)
        for k in ("excess_spk", "excess_sent", "acoustic_share"):
            row[k] = new[k]
    res["acoustic_share_method"] = METHOD
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    cfg = ap.parse_args()
    for f in cfg.files or sorted(RESULTS_DIR.glob("*.json")):
        res = json.loads(f.read_text())
        f.write_text(json.dumps(rescore(res), indent=1), encoding="utf-8")
        print(f"rescored {f}")


if __name__ == "__main__":
    sys.exit(main())
