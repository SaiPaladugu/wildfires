# Wildfire Ignition Cause Prediction

**Predicting whether Canadian wildfires are naturally caused (lightning) or human-caused using machine learning.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project develops a machine learning classification model to predict wildfire ignition cause in Canada. Using the Canadian National Fire Database (CNFDB) containing 14,485+ fires with labeled causes from 1972-2024, we train and evaluate multiple classifiers to distinguish between:

- **Natural fires** (N): Primarily lightning-caused (~86% of fires)
- **Human-caused fires** (H): Accidental, negligence, industrial (~14% of fires)

### Key Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 90.8% | 0.857 |
| Random Forest | 91.8% | 0.888 |
| **Gradient Boosting** | **91.6%** | **0.891** |

**Top Predictive Features:**
1. Day of Year (49%) - Natural fires peak in July-August (lightning season)
2. Fire Size (21%) - Natural fires tend to grow larger
3. Year (7%) - Temporal trends in fire causation
4. Region (7%) - Provincial variation in human-caused rates

## Repository Structure

```
wildfires/
├── README.md                 # This file
├── proposal/
│   └── proposal.tex          # LaTeX project proposal (ACM format)
├── src/
│   └── wildfire_cause_prediction.py  # Main analysis script
├── exploratory_analysis.py   # Initial data exploration
├── requirements.txt          # Python dependencies
└── .gitignore
```

## Data Sources

### Canadian National Fire Database (CNFDB)
- **Source**: [Natural Resources Canada - CWFIS Datamart](https://cwfis.cfs.nrcan.gc.ca/datamart)
- **Coverage**: 1972-2024, all Canadian provinces and territories
- **Download**: Fire polygon data from the link above
- **Place in**: Project root directory

### Environment Canada Climate Data
- **Source**: [Government of Canada](https://dd.weather.gc.ca/)
- **Coverage**: Daily weather observations from stations nationwide
- **Variables**: Temperature, precipitation, snow, wind

> **Note**: Data files are not included in this repository due to size. Download from the sources above.

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/SaiPaladugu/wildfires.git
cd wildfires
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the fire data
1. Go to [CWFIS Datamart](https://cwfis.cfs.nrcan.gc.ca/datamart)
2. Download "National Fire Database fire polygon data"
3. Extract the shapefiles to the project root

### 5. Download the climate data
1. Go to the [Climate Data Extraction Tool](https://climate-change.canada.ca/climate-data/#/daily-climate-data)
2. Go to "Technical information and metadata" to download the list of detailed information for each Daily climate station
3. Run filter_stations.py to generate the station list
4. Run download_climate_data.sh to download the daily climate data

## Running the Experiment

### Quick Start
```bash
# Activate virtual environment
source venv/bin/activate

# Run the main analysis
python src/wildfire_cause_prediction.py
```

### Output
The script will:
1. Load and preprocess the CNFDB fire data
2. Engineer temporal, geographic, and fire-based features
3. Train Logistic Regression, Random Forest, and Gradient Boosting classifiers
4. Evaluate models and display performance metrics
5. Generate visualizations in `outputs/`

### Exploratory Analysis
```bash
python exploratory_analysis.py
```
Generates initial data exploration and POC results in `analysis_output/`.

## Methodology

### Feature Engineering
- **Temporal**: Day of year, month, year, weekend indicator
- **Geographic**: Province/territory encoding
- **Fire characteristics**: Log-transformed fire size
- **Climate** (future): Temperature anomalies, precipitation deficit

### Models
- **Logistic Regression**: Interpretable baseline
- **Random Forest**: Handles non-linear relationships
- **Gradient Boosting**: Best overall performance

### Evaluation
- Stratified train/test split (80/20)
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Feature importance analysis

## Project Context

This project was developed for **COMP 4107/5107 - Data Science** at Carleton University.

### Research Question
> Can we accurately predict whether a wildfire was naturally or human-caused based on temporal, geographic, and climate features?

### Implications
- **Fire Prevention**: Target awareness campaigns to high-risk human-cause periods
- **Resource Allocation**: Optimize firefighting deployment strategies
- **Policy**: Evidence-based regulations for fire-prone activities

## References

1. Canadian Forest Service. (2025). Canadian National Fire Database. Natural Resources Canada. https://cwfis.cfs.nrcan.gc.ca/datamart

2. Environment and Climate Change Canada. Historical Climate Data. https://dd.weather.gc.ca/

3. Flannigan, M.D., et al. (2009). Implications of changing climate for global wildland fire. *International Journal of Wildland Fire*, 18(5), 483-507.

## License

MIT License - See LICENSE file for details.

## Authors

- [Student Name 1] - Technical Expert
- [Student Name 2] - Domain Expert  
- [Student Name 3] - Domain Expert

Carleton University, Winter 2026
