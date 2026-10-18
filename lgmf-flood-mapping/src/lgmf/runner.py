"""One-call experiment runner (v8): train -> select -> evaluate -> log everything NB-16 needs.

A run lives in  <out_root>/runs_e<EPOCHS>/<run_id>/seed<seed>/  and contains
    config.json    the full LGMFConfig + model spec used (reproduces the run)
    history.json   per epoch: losses, scheduled probs, MEASURED regime shares/counts, val IoU per regime
    best.pt        EMA + raw weights at the selected epoch (written atomically)
    perchip.csv.gz TP/FP/FN/TN per chip x regime for val/test/bolivia (pool them to get any metric)
    result.json    written LAST: its existence marks the run as finished
Re-running a finished run is a no-op; a run directory without result.json (crashed session) is deleted and redone.
"""
import copy
import dataclasses
import gzip
import json
import os
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import LGMFConfig
from .ema import ModelEMA
from .engine import train_one_epoch, evaluate
from .audit import chip_counts, summarise
from .model import count_params

REGS = ("both", "s1_only", "s2_only")


def seed_all(sd):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.benchmark = False


def build_from_spec(spec, pretrained=True):
    """spec = dict(kind='concat'|'gated', arch=..., encoder=..., [film=True, gate=True])."""
    from .baselines import build_baseline, build_gated_unet
    if spec["kind"] == "gated":
        return build_gated_unet(arch=spec["arch"], encoder=spec["encoder"], out_ch=8,
                                film=spec.get("film", True), gate=spec.get("gate", True), pretrained=pretrained)
    return build_baseline(arch=spec["arch"], encoder=spec["encoder"], in_ch=8, pretrained=pretrained)


def ckpt_tag(spec):
    if spec["kind"] == "gated":
        return {"gated": {"out_ch": 8, "film": spec.get("film", True), "gate": spec.get("gate", True),
                          "arch": spec["arch"], "encoder": spec["encoder"]}}
    return {"baseline": {"in_ch": 8, "arch": spec["arch"], "encoder": spec["encoder"]}}


def run_dir(out_root, epochs, run_id, seed):
    return os.path.join(out_root, f"runs_e{epochs}", run_id, f"seed{seed}")


def is_done(out_root, epochs, run_id, seed):
    return os.path.exists(os.path.join(run_dir(out_root, epochs, run_id, seed), "result.json"))


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp); os.replace(tmp, path)


def _score(val, select):
    if select == "mean":
        return float(np.mean([val[r]["iou"] for r in REGS]))
    return float(val[select]["iou"])


def run_experiment(run_id, cfg, spec, train_ds, val_ds, eval_caches, out_root, device,
                   select="mean", pretrained=True, log=print):
    """train_ds / val_ds: RamDataset (train augmented). eval_caches: {'val','test','bolivia': RamDataset}.
    select: 'mean' (paper) or one regime ('s1_only', 's2_only', 'both') for a specialist."""
    D = run_dir(out_root, cfg.epochs, run_id, cfg.seed)
    rj = os.path.join(D, "result.json")
    if os.path.exists(rj):
        log(f"[skip] {run_id} seed{cfg.seed}"); return json.load(open(rj))
    if os.path.isdir(D): shutil.rmtree(D)          # crashed earlier: truncated files, start clean
    os.makedirs(D)
    cfg = copy.deepcopy(cfg); cfg.run_name = run_id; cfg.ckpt_dir = out_root
    json.dump({"cfg": dataclasses.asdict(cfg), "spec": spec, "select": select}, open(os.path.join(D, "config.json"), "w"), indent=1, default=str)

    seed_all(cfg.seed)
    tr = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda", drop_last=True)
    va = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = build_from_spec(spec, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None
    ema = ModelEMA(model, cfg.ema_decay); em = ema.module

    hist, best, best_ep, best_val, t0 = [], -1.0, -1, None, time.time()
    for epoch in range(cfg.epochs):
        tr_log = train_one_epoch(model, tr, opt, cfg, epoch, device, scaler=scaler, ema=ema)
        sched.step()
        val = evaluate(em, va, cfg, device)
        sc = _score(val, select)
        hist.append({"epoch": epoch, "loss": tr_log["loss"], "seg": tr_log["seg"], "cons": tr_log["cons"],
                     "probs": list(tr_log["probs"]), "counts": tr_log["counts"], "shares": list(tr_log["shares"]),
                     "cons_examples": tr_log["cons_examples"], "score": sc,
                     "val": {r: {"iou": val[r]["iou"], "flood_only_iou": val[r].get("flood_only_iou")} for r in REGS}})
        if sc > best:
            best, best_ep = sc, epoch
            best_val = {r: {"iou": val[r]["iou"], "f1": val[r]["f1"], "flood_only_iou": val[r].get("flood_only_iou")} for r in REGS}
            _atomic_save({"model": model.state_dict(), "ema": ema.module.state_dict(), "val": best_val, "epoch": epoch,
                          "cfg": dataclasses.asdict(cfg), **ckpt_tag(spec)}, os.path.join(D, "best.pt"))
    train_min = (time.time() - t0) / 60
    json.dump(hist, open(os.path.join(D, "history.json"), "w"))

    # ---- final evaluation of the SELECTED weights, per chip, on val / test / bolivia (zero-fill = training fill)
    ck = torch.load(os.path.join(D, "best.pt"), map_location="cpu", weights_only=False)
    fm = build_from_spec(spec, pretrained=False); fm.load_state_dict(ck["ema"]); fm = fm.to(device).eval()
    rows = []
    for split, cache in eval_caches.items():
        rows += chip_counts(fm, cache, dict(config=run_id, seed=cfg.seed, split=split), device, "zero")
    import pandas as pd
    df = pd.DataFrame(rows)
    with gzip.open(os.path.join(D, "perchip.csv.gz"), "wt") as f: df.to_csv(f, index=False)
    res = {"run_id": run_id, "seed": cfg.seed, "select": select, "best_epoch": best_ep, "best_score": best,
           "params_M": float(count_params(fm)), "train_minutes": train_min, "spec": spec,
           "val": summarise(rows, "val"), "test": summarise(rows, "test"), "bolivia": summarise(rows, "bolivia"),
           "measured_cumulative_shares": [float(x) for x in np.sum(np.array([h["counts"] for h in hist]), 0) / max(1, np.sum([h["counts"] for h in hist]))],
           "scheduled_cumulative_probs": [float(x) for x in np.mean([h["probs"] for h in hist], 0)]}
    json.dump(res, open(rj, "w"), indent=1)
    log(f"[done] {run_id:18s} seed{cfg.seed}  sel {best:.3f}@{best_ep}  test both {res['test']['both']['iou']:.3f} "
        f"s1 {res['test']['s1_only']['iou']:.3f} s2 {res['test']['s2_only']['iou']:.3f} | bolivia s1 {res['bolivia']['s1_only']['iou']:.3f}  ({train_min:.1f} min)")
    del model, fm, ema, ck
    if device.type == "cuda": torch.cuda.empty_cache()
    return res
