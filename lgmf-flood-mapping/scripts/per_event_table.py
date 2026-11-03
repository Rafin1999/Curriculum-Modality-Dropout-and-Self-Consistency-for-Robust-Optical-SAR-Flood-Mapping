"""Table 13 of the paper: test-partition water-class IoU by flood event.

Input : nb16_perchip_all.csv.gz (output of notebooks/revision/rev16, also in the
        checkpoint Zenodo record, 10.5281/zenodo.23568792, under tables/).
Output: per_event_table.csv and the table printed to the console.

IoU is pooled over the chips of one event (TP / (TP + FP + FN)), computed per seed,
then averaged over seeds 42, 1 and 2.
Usage : python per_event_table.py path/to/nb16_perchip_all.csv.gz
"""
import sys
import numpy as np
import pandas as pd

MODELS = {"seg_full_ref": "SegFormer-B0, recipe", "seg_plain": "SegFormer-B0, no recipe",
          "unet_full": "U-Net, recipe", "manet_full": "MANet, recipe"}
SEEDS = (42, 1, 2)


def iou(g):
    tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
    return tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan


def main(path):
    d = pd.read_csv(path)
    t = d[d.split == "test"]
    rows = []
    for ev, ge in t.groupby("event"):
        r = {"event": ev, "chips": ge.id.nunique()}
        for run in MODELS:
            for reg in ("both", "s1_only"):
                v = [iou(ge[(ge.config == run) & (ge.seed == s) & (ge.regime == reg)]) for s in SEEDS]
                r[f"{run}|{reg}"] = round(float(np.nanmean(v)), 3)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("event")
    cols = ["event", "chips", "seg_full_ref|both", "seg_full_ref|s1_only", "seg_plain|s1_only",
            "unet_full|s1_only", "manet_full|s1_only"]
    print(out[cols].to_string(index=False))
    out.to_csv("per_event_table.csv", index=False)
    # check: pooled over all test chips this must reproduce Table 2
    for run in MODELS:
        v = [iou(t[(t.config == run) & (t.seed == s) & (t.regime == "s1_only")]) for s in SEEDS]
        print(f"pooled test SAR-only IoU, {MODELS[run]}: {np.mean(v):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "nb16_perchip_all.csv.gz")
