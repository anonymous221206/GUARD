# Data

Nothing in this directory is committed.  Each script downloads one source into
`data/raw/` and prints where it landed.

| script | what it fetches | size | licence |
|---|---|---|---|
| `download_drugban.sh` | DrugBAN code + BindingDB/BioSNAP/Human with official random and cluster splits | ~40 MB | MIT (code), see upstream for data |
| `download_vilt.sh` | missing-aware-prompt host code + pre-trained ViLT backbone | ~500 MB | Apache-2.0 / MIT |
| `download_opportunity.sh` | OPPORTUNITY activity recognition | ~300 MB | CC BY 4.0 |

## Datasets that cannot be scripted

Three vision-language corpora need a manual step because they sit behind a
click-through licence:

* **UPMC Food-101** -- <https://visiir.isir.upmc.fr/explore> (mirror on Kaggle:
  `gianmarco96/upmcfood101`)
* **Hateful Memes** -- <https://hatefulmemeschallenge.com/> (mirrors on Kaggle;
  the labelled test set is `test_seen.jsonl`, not `test.jsonl`)
* **MM-IMDb** -- <https://archive.org/download/mmimdb/mmimdb.tar.gz>

Place them under `data/raw/` following the layout in the host repository's
`DATA.md`, then run `hosts/vilt_prompts.py --prepare`.

## Affective-computing hosts

MOSI, MOSEI, IEMOCAP and CH-SIMS are used through the released checkpoints of
MoMKE, TMDC, CMAD, IMDer and LNLN.  We do not redistribute them; follow each
project's instructions and point `hosts/dumps.py` at the resulting predictions.
