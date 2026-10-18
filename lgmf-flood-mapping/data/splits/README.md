# data/splits/

The paired file index and normalization statistics are **derived files**: they
list where each Sen1Floods11 chip lives on the machine that built them, so they
are regenerated rather than versioned here.

Files (written by `notebooks/01_data_audit.ipynb`; also exported by
`notebooks/revision/rev11_audit_and_release.ipynb` in the release bundle):

- `train_index.json`, `val_index.json`, `test_index.json`, `bolivia_index.json`
- `norm_stats.json` (per-channel mean and std, computed on the training split)

Split sizes used in the paper: train 250 / validation 89 / test 90 / Bolivia 15
(the official hand-labelled split has 252 training chips; two with more than
98 % no-data pixels are dropped).

Each `*_index.json` is a list of records:

```json
{"id": "Bolivia_103757", "s1": "<path>", "s2": "<path>", "label": "<path>", "jrc": "<path or null>"}
```

The `id` is `<Event>_<chip number>`; the event is the part before the first
underscore. Paths are absolute and point to your copy of Sen1Floods11.

**To obtain the files**
1. Download Sen1Floods11 and the official split CSVs (see the main README).
2. Run `notebooks/01_data_audit.ipynb` with the paths at the top pointed at your copy.
3. Or download the archived copy from the Zenodo release (DOI in the main README),
   then rewrite the paths to your local data location.
