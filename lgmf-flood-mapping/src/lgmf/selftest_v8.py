"""Preflight for lgmf v8:  python -m lgmf.selftest_v8   (about 1-2 min on CPU or GPU, no real data needed)

Builds a tiny synthetic Sen1Floods11-like dataset (64x64 GeoTIFFs), then checks: the RAM cache equals the
original dataset (including augmentation RNG), measured regime shares, flops_g side effects, partial-cloud masks,
per-chip pooling == engine.evaluate, a full run_experiment (train, select, per-chip eval, skip, crash-restart),
and the symmetric-consistency path.
"""
import json
import os
import shutil
import tempfile

import numpy as np
import torch


def _make_data(root, n=(6, 4, 4, 3), size=64, seed=0):
    import rasterio
    from rasterio.transform import from_origin
    rng = np.random.RandomState(seed)
    os.makedirs(root, exist_ok=True)
    names = ["train", "val", "test", "bolivia"]
    idx = {}
    for sp, k in zip(names, n):
        items = []
        for i in range(k):
            cid = f"{'Bolivia' if sp == 'bolivia' else 'Ghana'}_{sp}{i}"
            tr = from_origin(0, 0, 10, 10)
            def w(name, arr):
                p = os.path.join(root, f"{cid}_{name}.tif")
                with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=arr.shape[0], dtype="float32", transform=tr) as d:
                    d.write(arr.astype("float32"))
                return p
            yy, xx = np.mgrid[0:size, 0:size]
            water = ((yy - rng.randint(10, 50)) ** 2 + (xx - rng.randint(10, 50)) ** 2 < rng.randint(80, 400)).astype("float32")
            lab = water.copy(); lab[:4, :] = -1
            s1 = np.stack([-12 - 6 * water + rng.randn(size, size), -18 - 6 * water + rng.randn(size, size)])
            s1[:, 10:14, 10:14] = np.nan                      # non-finite pixels
            s2 = np.stack([2000 - 1500 * water + 50 * rng.randn(size, size) for _ in range(13)])
            items.append(dict(id=cid, s1=w("S1Hand", s1), s2=w("S2Hand", s2), label=w("LabelHand", lab[None]),
                              jrc=w("JRCWaterHand", (water * (rng.rand() > 0.5))[None])))
        idx[sp] = items
        json.dump(items, open(os.path.join(root, f"{sp}_index.json"), "w"))
    stats = dict(s1_mean=[-13.0, -19.0], s1_std=[3.0, 3.0], s2_mean=[1500.0] * 6, s2_std=[600.0] * 6)
    json.dump(stats, open(os.path.join(root, "norm_stats.json"), "w"))
    return stats


def run(workdir=None, device=None, epochs=2):
    from .config import LGMFConfig
    from .dataset import Sen1Floods11
    from .ramdata import RamDataset
    from .engine import train_one_epoch, evaluate
    from .runner import run_experiment, run_dir, build_from_spec
    from .audit import chip_counts, summarise
    from .cloud import cloud_mask, cloud_and_shadow, apply_cloud
    from .metrics import flops_g
    from .ema import ModelEMA
    from torch.utils.data import DataLoader

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    own = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="lgmf_selftest_")
    try:
        data = os.path.join(workdir, "data"); stats = _make_data(data)
        cfg = LGMFConfig(); cfg.img_size = 64; cfg.num_workers = 0; cfg.batch_size = 2; cfg.epochs = epochs
        cfg.s1_mean, cfg.s1_std, cfg.s2_mean, cfg.s2_std = stats["s1_mean"], stats["s1_std"], stats["s2_mean"], stats["s2_std"]
        cfg.loss_type = "focal_tversky"; cfg.amp = False; cfg.warmup_epochs = 1
        P = lambda s: os.path.join(data, f"{s}_index.json")

        # 1) RAM cache == original dataset, incl. augmentation RNG order
        ref = Sen1Floods11(cfg, P("train"), augment=True); ram = RamDataset(cfg, P("train"), augment=True)
        torch.manual_seed(5); a = [ref[i] for i in range(len(ref))]
        torch.manual_seed(5); b = [ram[i] for i in range(len(ram))]
        for x, y in zip(a, b):
            for k in ("s1", "s2", "label", "valid"): assert torch.equal(x[k], y[k]), k
        print("1) RamDataset == Sen1Floods11 (incl. augmentation RNG)  OK")

        # 2) measured regime shares
        spec = dict(kind="concat", arch="segformer", encoder="mit_b0")
        model = build_from_spec(spec, pretrained=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3); tr = DataLoader(ram, batch_size=2, shuffle=True, drop_last=True)
        c0 = LGMFConfig(**{**cfg.__dict__}); c0.use_curriculum = False; c0.fixed_regime_probs = (1.0, 0.0, 0.0); c0.consistency_weight = 0.5
        out = train_one_epoch(model, tr, opt, c0, 0, device)
        assert out["shares"] == (1.0, 0.0, 0.0) and out["cons_examples"] == out["counts"][0] == 6, out
        c1 = LGMFConfig(**{**cfg.__dict__}); c1.use_curriculum = False; c1.fixed_regime_probs = (0.0, 0.5, 0.5); c1.consistency_weight = 0.5
        out = train_one_epoch(model, tr, opt, c1, 0, device)
        assert out["counts"][0] == 0 and sum(out["counts"]) == 6 and out["cons_examples"] == 0, out
        c2 = LGMFConfig(**{**cfg.__dict__}); c2.use_curriculum = False; c2.fixed_regime_probs = (1.0, 0.0, 0.0); c2.consistency_symmetric = True
        out = train_one_epoch(model, tr, opt, c2, 0, device); assert out["cons"] > 0
        print("2) measured regime shares + symmetric consistency path  OK", out["shares"])

        # 3) flops_g does not touch the caller's model
        keys0 = set(model.state_dict().keys()); dev0 = next(model.parameters()).device
        _ = flops_g(model, cfg, device="cpu")
        assert set(model.state_dict().keys()) == keys0 and next(model.parameters()).device == dev0
        print("3) flops_g leaves model device/state untouched  OK")

        # 4) partial-cloud masks
        for cov in (0.0, 0.25, 0.5, 0.75, 1.0):
            m = cloud_mask(128, 128, cov, seed=3); assert abs(m.mean() - cov) < 1e-3, (cov, m.mean())
        assert not np.array_equal(cloud_mask(64, 64, 0.5, 1), cloud_mask(64, 64, 0.5, 2))
        assert np.array_equal(cloud_mask(64, 64, 0.5, 7), cloud_mask(64, 64, 0.5, 7))
        c, s = cloud_and_shadow(64, 64, 0.3, 1); assert not (c & s).any()
        s2 = torch.randn(2, 6, 64, 64); mz = apply_cloud(s2, np.stack([c, c]), mode="zero")
        assert (mz[:, :, torch.from_numpy(c)] == 0).all() and torch.equal(mz[:, :, torch.from_numpy(~c)], s2[:, :, torch.from_numpy(~c)])
        mb = apply_cloud(s2, np.stack([c, c]), np.stack([s, s]), "bright", stats["s2_mean"], stats["s2_std"]); assert mb.shape == s2.shape
        print("4) cloud masks (exact coverage, deterministic, zero/bright)  OK")

        # 5) per-chip pooling == engine.evaluate;  cloud=0% == no cloud
        cache = RamDataset(cfg, P("test"), augment=False, keep_masks=True)
        em = ModelEMA(model, 0.9).module.to(device)
        ref_eval = evaluate(em, DataLoader(cache, batch_size=2), cfg, device)
        rows = chip_counts(em, cache, dict(config="t", seed=0, split="test"), device, "zero", bs=2)
        sm = summarise(rows, "test")
        for r in ("both", "s1_only", "s2_only"):
            assert abs(sm[r]["iou"] - ref_eval[r]["iou"]) < 1e-6 and abs(sm[r]["flood_iou"] - ref_eval[r]["flood_only_iou"]) < 1e-6, r
        rows0 = chip_counts(em, cache, dict(config="t", seed=0, split="test"), device, "zero", bs=2, cloud=dict(coverage=0.0))
        assert summarise(rows0, "test") == sm
        rows75 = chip_counts(em, cache, dict(config="t", seed=0, split="test"), device, "zero", bs=2, cloud=dict(coverage=0.75, seed=1))
        s75 = summarise(rows75, "test"); assert abs(s75["s1_only"]["iou"] - sm["s1_only"]["iou"]) < 1e-9   # SAR-only unaffected by optical clouds
        rowsm = chip_counts(em, cache, dict(config="t", seed=0, split="test"), device, "mean", bs=2); assert len(rowsm) == len(rows)
        print("5) per-chip pooling == engine.evaluate; clouds only touch optical regimes  OK")

        # 6) full run_experiment, skip, crash-restart
        caches = {s: RamDataset(cfg, P(s), augment=False, keep_masks=False) for s in ("val", "test", "bolivia")}
        val_ds = caches["val"]; out_root = os.path.join(workdir, "out")
        cfg.seed = 7; cfg.use_curriculum = True; cfg.consistency_weight = 0.5
        res = run_experiment("st_full", cfg, spec, RamDataset(cfg, P("train"), augment=True), val_ds, caches, out_root, device, pretrained=False, log=lambda *a: None)
        D = run_dir(out_root, epochs, "st_full", 7)
        for f in ("config.json", "history.json", "best.pt", "perchip.csv.gz", "result.json"): assert os.path.exists(os.path.join(D, f)), f
        h = json.load(open(os.path.join(D, "history.json"))); assert len(h) == epochs and "shares" in h[0]
        assert len(res["bolivia"]) == 3 and res["params_M"] > 3
        res2 = run_experiment("st_full", cfg, spec, None, None, caches, out_root, device, log=lambda *a: None); assert res2["best_epoch"] == res["best_epoch"]
        os.remove(os.path.join(D, "result.json"))                                           # simulate a crash
        res3 = run_experiment("st_full", cfg, spec, RamDataset(cfg, P("train"), augment=True), val_ds, caches, out_root, device, pretrained=False, log=lambda *a: None)
        assert os.path.exists(os.path.join(D, "result.json")) and abs(res3["test"]["both"]["iou"] - res["test"]["both"]["iou"]) < 0.5
        # specialist selection criterion
        res4 = run_experiment("st_sar", cfg, spec, RamDataset(cfg, P("train"), augment=True), val_ds, caches, out_root, device, select="s1_only", pretrained=False, log=lambda *a: None)
        assert res4["select"] == "s1_only"
        # checkpoint rebuild contract (what NB-11 / NB-16 do)
        from .runner import build_from_spec as b2
        ck = torch.load(os.path.join(D, "best.pt"), map_location="cpu", weights_only=False); assert ck["baseline"]["arch"] == "segformer"
        b2(spec, pretrained=False).load_state_dict(ck["ema"])
        gspec = dict(kind="gated", arch="unet", encoder="mobilenet_v2", film=True, gate=True)
        res5 = run_experiment("st_gated", cfg, gspec, RamDataset(cfg, P("train"), augment=True), val_ds, caches, out_root, device, pretrained=False, log=lambda *a: None)
        ckg = torch.load(os.path.join(run_dir(out_root, epochs, "st_gated", 7), "best.pt"), map_location="cpu", weights_only=False); assert ckg["gated"]["arch"] == "unet"
        print("6) run_experiment: files, skip, crash-restart, specialist select, gated spec  OK")
        print("\nlgmf v8 selftest PASSED")
    finally:
        if own: shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    run()
