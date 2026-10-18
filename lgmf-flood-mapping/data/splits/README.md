# data/splits/

This folder holds the exact split used in the paper. The index files were
written by `notebooks/01_data_audit.ipynb` on Kaggle.

Files:

- `train_index.json`, `val_index.json`, `test_index.json`, `bolivia_index.json`
- `norm_stats.json` (per-channel mean and std, computed on the training split)

- `official_csv/flood_{train,valid,test,bolivia}_data.csv`: the official
  Sen1Floods11 hand-labelled split lists (252 / 89 / 90 / 15 chips), unchanged.

Split sizes used in the paper: train 250 / validation 89 / test 90 / Bolivia 15.
`norm_stats.json` records the two training chips that were dropped because more
than 98 % of their pixels are no-data (Ghana_161233, Ghana_26376) and the
Sentinel-2 bands used (indices 1, 2, 3, 7, 11, 12).

Each `*_index.json` is a list of records:

```json
{"id": "Bolivia_103757", "s1": "<path>", "s2": "<path>", "label": "<path>", "jrc": "<path or null>"}
```

The `id` is `<Event>_<chip number>`; the event is the part before the first
underscore.

**Paths.** The stored paths are the Kaggle paths used for the experiments
(`/kaggle/input/datasets/robertomarinoformica/sen1floods11-dataset/HandLabeled/...`),
so the notebooks run unchanged if you attach that public Kaggle copy of
Sen1Floods11. On another machine, replace the path prefix with the location of
your own copy; the file names (`<id>_S1Hand.tif`, `_S2Hand.tif`, `_LabelHand.tif`,
`_JRCWaterHand.tif`) follow the official release.
