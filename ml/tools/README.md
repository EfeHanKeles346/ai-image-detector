# Dataset tools

Acquisition and auditing for the 255 GB of evaluation data described in `HISTORY.md` §1c.
These live in the repo rather than beside the data, because the *auditing rule* is part of the
method — see `HISTORY.md` §1b for why.

## `audit_datasets.py`

Runs five mechanical checks on any dataset folder and writes a verdict. The checks all answer one
question: **could a model separate the classes without looking at image content?**

1. format split (JPEG vs PNG — this is what `archive1` had)
2. shape split (all AI square, all real rectangular)
3. resolution split (median side ratio > 2.5)
4. compression split (bytes per pixel)
5. class balance

Reads inside parquet, zip and tar without extracting anything.

```bash
.venv/bin/python tools/audit_datasets.py                 # everything on the SSD
.venv/bin/python tools/audit_datasets.py <folder>        # one dataset
```

Two sampling details it gets right, both learned the hard way:
- skips macOS AppleDouble stubs (`._name`), which ExFAT writes beside every real file
- samples across the **whole** shard range, because some datasets are sorted by label and reading
  the first few shards reports "single class" for a balanced set

## `fetch_datasets.py` + `watchdog.sh`

Unattended downloader, built after the first attempt died 31 minutes in. Four independent layers:
one subprocess per dataset (a poisoned HTTP client cannot cascade), exponential backoff with
longer waits on 429, multiple passes over the queue, and a shell watchdog outside the process.

```bash
./tools/watchdog.sh            # runs until every dataset is done
```

Edit `QUEUE` in `fetch_datasets.py` to change what gets downloaded. Order it by **file count**,
not size: a dataset of 8,000 small files costs 8,000 API calls and will trigger rate limiting long
before a 260 GB dataset made of 188 large ones.

**Known gap:** these download first and audit second. The right order is to pull one shard, audit
it, and only then commit to the full download — recorded in `HISTORY.md` §2b.8.

## E32 authentic-photo acquisition

`experiments/e32_data_system.py` is the role-safe replacement for adding the E32 camera sources.
Its default operation freezes and verifies metadata only; every image transfer is explicit. The
exact receipt and third-party bytes remain below `$PIXELPROOF_DATA_ROOT/e32`.

```bash
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
  PYTHONPATH=src .venv/bin/python experiments/e32_data_system.py freeze-real
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
  PYTHONPATH=src .venv/bin/python experiments/e32_data_system.py download-real --source vision
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
  PYTHONPATH=src .venv/bin/python experiments/e32_data_system.py status
```

Valid source names are `vision`, `fodb` and `csafe`. Downloads use `.partial` files, retry/resume,
declared archive sizes and a 100 GiB free-space floor. A completed transfer is still not training
data until its decoded parent inventory and protected-content audit pass.

The separate AI-holdings inventory is read-only:

```bash
PYTHONPATH=src .venv/bin/python experiments/e32_ai_inventory.py \
  --root /Volumes/LaCie/pixelproof-datasets \
  --output ../evidence/e32_ai_inventory.json
```

It counts Parquet rows, loose image/sidecar pairs and ZIP image members, but deliberately refuses
to infer missing dataset licences or to move protected test sources into training.

The licensed two-family gap selection is also metadata-first:

```bash
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
  PYTHONPATH=src:experiments .venv/bin/python experiments/e32_gap_acquisition.py freeze
PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
  PYTHONPATH=src:experiments .venv/bin/python experiments/e32_gap_acquisition.py download \
  --source qwen-image-2512 --smoke
```

Run one `--smoke` for both `qwen-image-2512` and `flux2-klein-9b`, verify JPEG XL decode, then
remove `--smoke` for the frozen bulk. This sequence is a scientific role gate, not merely a download
optimization.

## Paths

Both scripts read `PIXELPROOF_DATA_ROOT`; the portable default is `ml/data/`. For an external
volume, set it explicitly before either command, for example:

```bash
export PIXELPROOF_DATA_ROOT=/path/to/pixelproof-datasets
```

`watchdog.sh` uses `ml/.venv/bin/python` by default. `PIXELPROOF_PYTHON` can select another
interpreter without editing the script.
