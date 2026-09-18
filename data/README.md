# Data

Raw datasets and processed arrays are not committed. Each script downloads or
checks one source and prints where it landed.

| script | what it fetches | size | licence |
|---|---|---|---|
| `download_artifacts.sh` | saved frozen-host outputs/checkpoints | ~2.9 GB | derived artifacts; upstream datasets retain their licences |
| `download_drugban.sh` | DrugBAN code and released data | ~40 MB | MIT (code), see upstream for data |
| `download_opportunity.sh` | OPPORTUNITY activity recognition | ~300 MB | CC BY 4.0 |

## Sources requiring a manual licence step

IEMOCAP, NinaPro DB5, CMU-MOSEI and AVE must be obtained under their original
licences. The scripts under `scripts/download/` document the expected layout
and continue with preprocessing after the source data are installed.
