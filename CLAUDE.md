# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML classification project predicting Canadian wildfire ignition cause (natural/lightning vs human-caused) using the Canadian National Fire Database (CNFDB). Built for COMP 4107/5107 - Data Science at Carleton University.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run main analysis (trains Logistic Regression, Random Forest, Gradient Boosting)
python src/wildfire_cause_prediction.py

# Run exploratory analysis (POC evaluation of multiple thesis topics)
python exploratory_analysis.py
```

No test suite or linter is configured.

## Architecture

Two standalone scripts, no shared modules:

- **`src/wildfire_cause_prediction.py`** — Main pipeline: loads NFDB fire data + climate CSVs, preprocesses, engineers features (DOY, cyclic encoding, region, log fire size, decade, weekend indicator), trains 3 classifiers (Logistic Regression, Random Forest, Gradient Boosting), evaluates with accuracy/ROC-AUC/confusion matrix, saves plots to `outputs/`.

- **`exploratory_analysis.py`** — Initial EDA: evaluates 4 POC topics (cause prediction, size prediction, temporal trends, regional analysis). Outputs to `analysis_output/`.

Both scripts use `BASE_PATH` relative to `data/` in the repo root.

## Data

Data files live in `data/` (gitignored). Downloaded via `download_climate_data.sh` and direct CWFIS URLs:
- **`data/NFDB_poly_large_fires/`** — Large fire (>=200ha) polygon shapefiles/DBF (16,394 records, 1972-2024)
- **`data/NFDB_poly/`** — All fire polygon shapefiles/DBF
- **`data/NFDB_point/`** — Fire point data with lat/lon (442,403 records)
- **`data/NFDB_point_txt/`** — Fire point data in text format
- **`data/NFDB_point_stats/`** — Summary statistics (Excel)
- **`data/climate/`** — Daily climate CSVs from 16 Environment Canada stations across 8 fire-prone provinces (BC, AB, SK, MB, ON, QC, NT, YT), 1972-2024, ~842 files
- **`data/Station_Inventory_EN.csv`** — Environment Canada station metadata

Key fire data columns: `YEAR`, `MONTH`, `DAY`, `CAUSE` (N/H/H-PB), `SIZE_HA`, `CALC_HA`, `SRC_AGENCY`.

## Course Requirements

This is a group project for COMP 4107/5107 - Data Science at Carleton University. Groups of 2-3 students: one technical expert (CS/Engineering/IT/Physics/Chemistry) and one or two domain experts (Communications, Geography, Biology, etc.). Technical experts build models; domain experts contribute problem framing, motivation, and implications.

### Deliverables

1. **Proposal** (done): Max 2-page PDF in ACM format. Includes problem statement, motivation, objectives. Located in `proposal/`.
2. **Presentation**: 20-minute conference-style talk on March 24 or April 7, plus 3-5 min Q&A. Structure: Introduction/motivation, research questions, methodology (data collection, cleanup, mining, analysis), results, implications, conclusion.
3. **Report** (due April 14, 11:59 PM): 8-10 pages, double column, ACM or IEEE format, submitted as PDF via email. Worth 50% of course grade.

## Key Design Decisions

- Binary classification: `CAUSE` values N (natural) vs H/H-PB (human) — ~86%/14% class imbalance
- Fire size is log-transformed (`np.log1p`) for normality
- Day-of-year gets cyclic encoding (`DOY_SIN`, `DOY_COS`) in the main script
- Stratified 80/20 train/test split with `random_state=42`
- Logistic Regression uses `StandardScaler`; tree-based models use raw features
