# data/splits/

This folder should contain the five files produced by
`notebooks/01_data_audit.ipynb`:

- `train_index.json`
- `val_index.json`
- `test_index.json`
- `bolivia_index.json`
- `norm_stats.json`

These were generated on Kaggle and saved as the `lgmf-index` dataset
(~5 MB total). **Before pushing this repo publicly**, copy those five files
from your Kaggle dataset output into this folder, replacing this README (or
keeping it alongside them).

Quickest way to fetch them if you no longer have a local copy:

```bash
kaggle datasets download -d <your-username>/lgmf-index -p ./data/splits --unzip
```

Each `*_index.json` is a list of records of the form:

```json
{"id": "Bolivia_103757", "s1": "<path>", "s2": "<path>", "label": "<path>", "jrc": "<path or null>"}
```

The `s1`/`s2`/`label`/`jrc` paths are absolute Kaggle paths and will not
resolve outside that environment. Update `dataset.py`'s expectations or
regenerate the index against your local Sen1Floods11 copy (rerun
`01_data_audit.ipynb` pointed at local paths) if you're not running on Kaggle.
