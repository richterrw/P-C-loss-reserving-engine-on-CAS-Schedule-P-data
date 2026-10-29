# Data

`raw/*_pos.csv` — CAS Loss Reserving Database (Meyers & Shi), NAIC Schedule P, one file per line:
wkcomp (Part D), ppauto (B), comauto (C), medmal (F2, claims-made), prodliab (R1), othliab (H1).
Accident years 1988–1997, development lags 1–10, net of reinsurance, $000s.
Original source: casact.org loss reserving data page; copied unmodified from the
`problemofpoints/reservetestr` repository (`data-raw/`).

`benchmarks/raa.csv`, `benchmarks/genins.csv` — the RAA (Mack 1993) and Taylor & Ashe (1983)
cumulative triangles, from the `casact/chainladder-python` sample data.

The MIT license in the repository root covers the code. The data files remain the CAS's and the
benchmark authors'; they are included unmodified for reproducibility.
