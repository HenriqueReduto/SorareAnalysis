# Data Preparation

Datasets are separated by provenance:

- `datasets/sorare/`: Sorare-only score exports.
- `datasets/transfermarkt/`: Transfermarkt-only source or prepared data.
- `datasets/combined/`: datasets created by joining Sorare and Transfermarkt data.

Temporary downloaded Transfermarkt tables are cached in `cache/transfermarkt/` and are ignored by git.
