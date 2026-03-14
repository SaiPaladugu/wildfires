"""
Wildfire Ignition Cause Prediction — Full Experiment
=====================================================
Binary classification: Natural (lightning) vs Human-caused fires
Using CNFDB fire data + Environment Canada daily climate data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dbfread import DBF
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    matthews_corrcoef, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# 1. LOAD FIRE DATA
# =============================================================================
print("=" * 70)
print("STEP 1: LOADING FIRE DATA")
print("=" * 70)

# Use point data — it has lat/lon for matching to weather stations
dbf_path = next(BASE_PATH.glob("NFDB_point/*.dbf"))
print(f"Loading: {dbf_path.name}")
table = DBF(str(dbf_path), encoding='utf-8', ignore_missing_memofile=True)
fire_df = pd.DataFrame(iter(table))
print(f"Total fire records: {len(fire_df):,}")
print(f"Columns: {list(fire_df.columns)}")

# Filter to fires with known cause (Natural or Human)
valid_causes = ['N', 'H', 'H-PB']
df = fire_df[fire_df['CAUSE'].isin(valid_causes)].copy()
df['target'] = df['CAUSE'].apply(lambda x: 0 if x == 'N' else 1)  # 0=Natural, 1=Human
print(f"\nFires with known cause: {len(df):,}")
print(f"  Natural (0): {(df['target']==0).sum():,} ({(df['target']==0).mean()*100:.1f}%)")
print(f"  Human   (1): {(df['target']==1).sum():,} ({(df['target']==1).mean()*100:.1f}%)")

# Clean numeric columns
for col in ['YEAR', 'MONTH', 'DAY', 'LATITUDE', 'LONGITUDE', 'SIZE_HA']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter to reasonable year range and valid coordinates
df = df[(df['YEAR'] >= 1972) & (df['YEAR'] <= 2024)].copy()
df = df[df['LATITUDE'].notna() & df['LONGITUDE'].notna()].copy()
df = df[(df['MONTH'] >= 1) & (df['MONTH'] <= 12)].copy()
print(f"After filtering (1972-2024, valid coords): {len(df):,}")

# =============================================================================
# 2. LOAD CLIMATE DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: LOADING CLIMATE DATA")
print("=" * 70)

climate_dir = BASE_PATH / "climate"
climate_files = sorted(climate_dir.glob("climate_daily_*.csv"))
print(f"Climate files: {len(climate_files)}")

climate_dfs = []
for f in climate_files:
    try:
        cdf = pd.read_csv(f, encoding='utf-8-sig')
        climate_dfs.append(cdf)
    except:
        pass

climate_df = pd.concat(climate_dfs, ignore_index=True)
print(f"Total climate observations: {len(climate_df):,}")

# Extract station info for spatial matching
stations = climate_df.groupby('Station Name').agg({
    'Longitude (x)': 'first',
    'Latitude (y)': 'first',
    'Climate ID': 'first'
}).reset_index()
stations.columns = ['station_name', 'stn_lon', 'stn_lat', 'climate_id']
print(f"Unique stations: {len(stations)}")
print(stations[['station_name', 'stn_lat', 'stn_lon']].to_string(index=False))

# Clean climate data
climate_df['Year'] = pd.to_numeric(climate_df['Year'], errors='coerce')
climate_df['Month'] = pd.to_numeric(climate_df['Month'], errors='coerce')
climate_df['Day'] = pd.to_numeric(climate_df['Day'], errors='coerce')

for col in ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)',
            'Total Rain (mm)', 'Total Snow (cm)', 'Total Precip (mm)',
            'Spd of Max Gust (km/h)', 'Snow on Grnd (cm)']:
    if col in climate_df.columns:
        climate_df[col] = pd.to_numeric(climate_df[col], errors='coerce')

# =============================================================================
# 3. SPATIAL MATCHING — FIRE TO NEAREST WEATHER STATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: MATCHING FIRES TO NEAREST WEATHER STATIONS")
print("=" * 70)

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# For each fire, find nearest station
stn_lats = stations['stn_lat'].values.astype(float)
stn_lons = stations['stn_lon'].values.astype(float)
stn_names = stations['station_name'].values

fire_lats = df['LATITUDE'].values
fire_lons = df['LONGITUDE'].values

nearest_station = []
nearest_dist = []

for i in range(len(df)):
    dists = haversine_km(fire_lats[i], fire_lons[i], stn_lats, stn_lons)
    idx = np.argmin(dists)
    nearest_station.append(stn_names[idx])
    nearest_dist.append(dists[idx])

df['nearest_station'] = nearest_station
df['station_dist_km'] = nearest_dist

print(f"Median fire-to-station distance: {np.median(nearest_dist):.0f} km")
print(f"90th percentile distance: {np.percentile(nearest_dist, 90):.0f} km")
print(f"\nFires per station:")
print(df['nearest_station'].value_counts().to_string())

# =============================================================================
# 4. JOIN CLIMATE DATA TO FIRES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: JOINING CLIMATE DATA TO FIRES")
print("=" * 70)

# Check for cached joined data
cache_path = BASE_PATH / "joined_fire_climate.parquet"
if cache_path.exists():
    print(f"Loading cached data from {cache_path.name}...")
    df = pd.read_parquet(cache_path)
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")

else:
    # Pre-compute climate features per station/year/month/day
    # We want: conditions on the day of fire + antecedent conditions (prior 7 and 30 days)

    # Create a date column for climate data
    climate_df['date'] = pd.to_datetime(
        climate_df[['Year', 'Month', 'Day']].rename(
            columns={'Year': 'year', 'Month': 'month', 'Day': 'day'}
        ), errors='coerce'
    )

    # Index climate data for fast lookup
    climate_df = climate_df.dropna(subset=['date'])
    climate_indexed = climate_df.set_index(['Station Name', 'date']).sort_index()

    def get_climate_features(station, year, month, day):
        """Extract climate features for a fire: day-of, 7-day, and 30-day antecedent."""
        try:
            fire_date = pd.Timestamp(year=int(year), month=int(month), day=int(day))
        except:
            return {}

        features = {}

        try:
            stn_data = climate_indexed.loc[station]
        except KeyError:
            return features

        # Day-of-fire conditions
        if fire_date in stn_data.index:
            row = stn_data.loc[fire_date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            features['temp_max'] = row.get('Max Temp (°C)', np.nan)
            features['temp_min'] = row.get('Min Temp (°C)', np.nan)
            features['temp_mean'] = row.get('Mean Temp (°C)', np.nan)
            features['precip'] = row.get('Total Precip (mm)', np.nan)
            features['rain'] = row.get('Total Rain (mm)', np.nan)
            features['snow_ground'] = row.get('Snow on Grnd (cm)', np.nan)
            features['wind_gust'] = row.get('Spd of Max Gust (km/h)', np.nan)

        # Antecedent conditions: 7-day and 30-day windows before fire
        for window, label in [(7, '7d'), (30, '30d')]:
            start = fire_date - pd.Timedelta(days=window)
            try:
                window_data = stn_data.loc[start:fire_date - pd.Timedelta(days=1)]
                if len(window_data) >= window // 2:  # Need at least half the window
                    features[f'temp_mean_{label}'] = window_data['Mean Temp (°C)'].mean()
                    features[f'temp_max_{label}'] = window_data['Max Temp (°C)'].max()
                    features[f'precip_total_{label}'] = window_data['Total Precip (mm)'].sum()
                    features[f'rain_total_{label}'] = window_data['Total Rain (mm)'].sum()
                    features[f'dry_days_{label}'] = (window_data['Total Precip (mm)'].fillna(0) == 0).sum()
                    features[f'temp_range_{label}'] = (
                        window_data['Max Temp (°C)'].max() - window_data['Min Temp (°C)'].min()
                    )
            except:
                pass

        return features

    print("Extracting climate features for each fire (this may take a few minutes)...")

    # Process in batches for progress reporting
    n = len(df)
    climate_features_list = []
    batch_size = 10000

    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch = df.iloc[start_idx:end_idx]

        for _, row in batch.iterrows():
            feats = get_climate_features(
                row['nearest_station'], row['YEAR'], row['MONTH'], row['DAY']
            )
            climate_features_list.append(feats)

        print(f"  Processed {end_idx:,}/{n:,} fires...")

    climate_feat_df = pd.DataFrame(climate_features_list, index=df.index)
    df = pd.concat([df, climate_feat_df], axis=1)

    # Report coverage
    for col in ['temp_mean', 'precip', 'temp_mean_7d', 'precip_total_30d']:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col}: {pct:.1f}% coverage")

    # Cache for faster re-runs
    df.to_parquet(cache_path)
    print(f"  Cached joined data to {cache_path.name}")

# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 70)

# --- Temporal features ---
# Day of year with cyclic encoding (captures circular seasonality)
def safe_doy(row):
    try:
        return pd.Timestamp(year=2000, month=int(row['MONTH']), day=int(row['DAY'])).dayofyear
    except:
        return np.nan

df['doy'] = df.apply(safe_doy, axis=1)
df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)

# Day of week (human fires may cluster on weekends)
def safe_dow(row):
    try:
        return pd.Timestamp(
            year=int(row['YEAR']), month=int(row['MONTH']), day=int(row['DAY'])
        ).dayofweek
    except:
        return np.nan

df['dow'] = df.apply(safe_dow, axis=1)
df['is_weekend'] = (df['dow'] >= 5).astype(float)

# Month as numeric
df['month'] = df['MONTH']

# Year (captures long-term trends in human activity)
df['year'] = df['YEAR']

# --- Geographic features ---
# Province as categorical code
province_map = {v: i for i, v in enumerate(sorted(df['SRC_AGENCY'].dropna().unique()))}
df['province_code'] = df['SRC_AGENCY'].map(province_map)

# Latitude and longitude (proxy for remoteness, vegetation, population)
df['latitude'] = df['LATITUDE']
df['longitude'] = df['LONGITUDE']

# --- Fire characteristics ---
df['log_size'] = np.log1p(df['SIZE_HA'].clip(lower=0))

# --- Climate-derived features ---
# Diurnal temperature range (day of fire) — proxy for clear/dry conditions
if 'temp_max' in df.columns and 'temp_min' in df.columns:
    df['diurnal_range'] = df['temp_max'] - df['temp_min']

# Consecutive dry days in prior 30 days — drought proxy
# (already computed as dry_days_30d)

# Temperature anomaly: how much hotter is the fire day vs. 30-day average?
if 'temp_mean' in df.columns and 'temp_mean_30d' in df.columns:
    df['temp_anomaly'] = df['temp_mean'] - df['temp_mean_30d']

# Precipitation deficit: 30-day precip below expected
# (using raw precip_total_30d — low values indicate drought)

# Fire weather proxy: hot + dry + windy
if all(c in df.columns for c in ['temp_max', 'precip', 'wind_gust']):
    df['fire_weather_index'] = (
        df['temp_max'].fillna(0) -
        df['precip'].fillna(0) * 2 +
        df['wind_gust'].fillna(0) * 0.5
    )

# Station distance (fires far from stations = more remote = more likely natural)
df['station_dist'] = df['station_dist_km']

# --- Define final feature set ---
feature_cols = [
    # Temporal
    'doy', 'doy_sin', 'doy_cos', 'month', 'year', 'is_weekend',
    # Geographic
    'province_code', 'latitude', 'longitude',
    # Fire
    'log_size',
    # Climate — day of fire
    'temp_max', 'temp_min', 'temp_mean', 'precip', 'rain',
    'snow_ground', 'wind_gust', 'diurnal_range',
    # Climate — antecedent 7-day
    'temp_mean_7d', 'temp_max_7d', 'precip_total_7d', 'rain_total_7d',
    'dry_days_7d', 'temp_range_7d',
    # Climate — antecedent 30-day
    'temp_mean_30d', 'temp_max_30d', 'precip_total_30d', 'rain_total_30d',
    'dry_days_30d', 'temp_range_30d',
    # Derived
    'temp_anomaly', 'fire_weather_index', 'station_dist',
]

# Only keep features that exist
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"Feature columns ({len(feature_cols)}):")
for c in feature_cols:
    non_null = df[c].notna().sum()
    print(f"  {c:<25} {non_null:>8,} non-null ({non_null/len(df)*100:.1f}%)")

# =============================================================================
# 6. PREPARE MODELING DATASET
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: PREPARING MODELING DATASET")
print("=" * 70)

# For GBDT, we can tolerate some missing values, but let's be selective
# Require core features: temporal + geographic + fire size
core_features = ['doy', 'province_code', 'latitude', 'longitude', 'log_size', 'year']
model_df = df.dropna(subset=core_features + ['target']).copy()
print(f"Samples with core features: {len(model_df):,}")

X = model_df[feature_cols].copy()
y = model_df['target'].values

# Fill remaining NaN with -999 (GBDT handles this natively as missing indicator)
X = X.fillna(-999)

print(f"Final dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"Class distribution: Natural={sum(y==0):,} ({sum(y==0)/len(y)*100:.1f}%), "
      f"Human={sum(y==1):,} ({sum(y==1)/len(y)*100:.1f}%)")

# Train/test split (stratified, 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# =============================================================================
# 7. TRAIN GRADIENT BOOSTED DECISION TREE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: TRAINING GBDT")
print("=" * 70)

# Use class weights to handle imbalance
# Scale_pos_weight: ratio of negative to positive samples
n_neg = sum(y_train == 0)
n_pos = sum(y_train == 1)
scale_ratio = n_neg / n_pos
print(f"Class imbalance ratio: {scale_ratio:.1f}:1 (Natural:Human)")

# sklearn GradientBoostingClassifier with sample weights
sample_weights = np.where(y_train == 1, scale_ratio, 1.0)

gbdt = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=20,
    subsample=0.8,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=1e-4,
)

print("Training GBDT (500 trees, lr=0.05, depth=5)...")
gbdt.fit(X_train, y_train, sample_weight=sample_weights)
print(f"Stopped at {gbdt.n_estimators_} estimators")

# =============================================================================
# 8. COMPREHENSIVE EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: EVALUATION RESULTS")
print("=" * 70)

y_pred = gbdt.predict(X_test)
y_prob = gbdt.predict_proba(X_test)[:, 1]

# --- Classification Report ---
print("\n[8.1] Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Natural', 'Human'], digits=3))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"[8.2] Confusion Matrix:")
print(f"                Predicted")
print(f"              Natural  Human")
print(f"  Actual Nat  {tn:>7,}  {fp:>5,}")
print(f"  Actual Hum  {fn:>7,}  {tp:>5,}")
print()

# --- Key Metrics ---
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision_h = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_h = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_h = 2 * precision_h * recall_h / (precision_h + recall_h) if (precision_h + recall_h) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)
brier = brier_score_loss(y_test, y_prob)
logloss = log_loss(y_test, y_prob)

print(f"[8.3] Key Metrics Summary:")
print(f"  {'Metric':<30} {'Value':>10}")
print(f"  {'-'*42}")
print(f"  {'Accuracy':<30} {accuracy:>10.4f}")
print(f"  {'ROC-AUC':<30} {roc_auc:>10.4f}")
print(f"  {'PR-AUC (Avg Precision)':<30} {pr_auc:>10.4f}")
print(f"  {'F1-Score (Human class)':<30} {f1_h:>10.4f}")
print(f"  {'Precision (Human class)':<30} {precision_h:>10.4f}")
print(f"  {'Recall (Human class)':<30} {recall_h:>10.4f}")
print(f"  {'Specificity (Natural class)':<30} {specificity:>10.4f}")
print(f"  {'Matthews Corr. Coef. (MCC)':<30} {mcc:>10.4f}")
print(f"  {'Brier Score':<30} {brier:>10.4f}")
print(f"  {'Log Loss':<30} {logloss:>10.4f}")
print()

# --- Baseline comparison (always predict majority class) ---
baseline_acc = max(y_test.mean(), 1 - y_test.mean())
# Random classifier PR-AUC = prevalence
baseline_pr_auc = y_test.mean()
print(f"[8.4] Baseline Comparisons:")
print(f"  Majority-class accuracy:   {baseline_acc:.4f}")
print(f"  Random-classifier PR-AUC:  {baseline_pr_auc:.4f}")
print(f"  Our accuracy improvement:  +{accuracy - baseline_acc:.4f} ({(accuracy - baseline_acc)/baseline_acc*100:.1f}%)")
print(f"  Our PR-AUC improvement:    +{pr_auc - baseline_pr_auc:.4f} ({(pr_auc - baseline_pr_auc)/baseline_pr_auc*100:.1f}%)")

# =============================================================================
# 9. STATISTICAL SIGNIFICANCE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: STATISTICAL SIGNIFICANCE")
print("=" * 70)

# 5-fold stratified cross-validation
print("\n[9.1] 5-Fold Stratified Cross-Validation:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# CV without sample weights (scoring is on held-out fold, weights affect fit only marginally)
cv_results = cross_validate(
    gbdt, X, y,
    cv=cv,
    scoring=['roc_auc', 'average_precision', 'f1', 'accuracy'],
    n_jobs=-1,
    return_train_score=False
)

for metric in ['roc_auc', 'average_precision', 'f1', 'accuracy']:
    scores = cv_results[f'test_{metric}']
    print(f"  {metric:<25} mean={scores.mean():.4f}  std={scores.std():.4f}  "
          f"[{scores.min():.4f}, {scores.max():.4f}]")

# Permutation test: is the model significantly better than chance?
print("\n[9.2] Permutation Test (1000 iterations):")
n_permutations = 1000
observed_auc = roc_auc
perm_aucs = []

rng = np.random.RandomState(RANDOM_STATE)
for _ in range(n_permutations):
    perm_y = rng.permutation(y_test)
    perm_auc = roc_auc_score(perm_y, y_prob)
    perm_aucs.append(perm_auc)

perm_aucs = np.array(perm_aucs)
p_value = (perm_aucs >= observed_auc).mean()

print(f"  Observed ROC-AUC:   {observed_auc:.4f}")
print(f"  Permutation mean:   {perm_aucs.mean():.4f} (std={perm_aucs.std():.4f})")
print(f"  p-value:            {p_value:.6f}")
print(f"  Significant at 0.001: {'YES' if p_value < 0.001 else 'NO'}")

# Bootstrap confidence intervals
print("\n[9.3] Bootstrap 95% Confidence Intervals (1000 resamples):")
n_bootstrap = 1000
boot_metrics = {'roc_auc': [], 'pr_auc': [], 'f1': [], 'accuracy': []}

for _ in range(n_bootstrap):
    idx = rng.choice(len(y_test), size=len(y_test), replace=True)
    by = y_test[idx]
    bp = y_prob[idx]
    bpred = y_pred[idx]

    if len(np.unique(by)) < 2:
        continue
    boot_metrics['roc_auc'].append(roc_auc_score(by, bp))
    boot_metrics['pr_auc'].append(average_precision_score(by, bp))
    boot_metrics['f1'].append(f1_score(by, bpred))
    boot_metrics['accuracy'].append((by == bpred).mean())

for metric, values in boot_metrics.items():
    values = np.array(values)
    lo, hi = np.percentile(values, [2.5, 97.5])
    print(f"  {metric:<15} {values.mean():.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

# =============================================================================
# 10. FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': gbdt.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
print(f"  {'Feature':<30} {'Importance':>10} {'Cumulative':>10}")
print(f"  {'-'*52}")
cum = 0
for _, row in importance.head(20).iterrows():
    cum += row['importance']
    print(f"  {row['feature']:<30} {row['importance']:>10.4f} {cum:>10.4f}")

# =============================================================================
# 11. SAVE VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11: SAVING VISUALIZATIONS")
print("=" * 70)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Wildfire Ignition Cause Prediction — GBDT Results', fontsize=14, fontweight='bold')

    # 1. ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'GBDT (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()

    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec, prec, 'r-', lw=2, label=f'GBDT (PR-AUC={pr_auc:.3f})')
    ax.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.5, label=f'Baseline={y_test.mean():.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()

    # 3. Confusion Matrix
    ax = axes[0, 2]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Natural', 'Human'], yticklabels=['Natural', 'Human'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # 4. Feature Importance (top 15)
    ax = axes[1, 0]
    top_imp = importance.head(15).sort_values('importance')
    ax.barh(top_imp['feature'], top_imp['importance'], color='teal')
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances')

    # 5. Calibration curve
    ax = axes[1, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, 's-', label='GBDT')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()

    # 6. Score distribution by class
    ax = axes[1, 2]
    ax.hist(y_prob[y_test == 0], bins=50, alpha=0.6, label='Natural', color='forestgreen', density=True)
    ax.hist(y_prob[y_test == 1], bins=50, alpha=0.6, label='Human', color='coral', density=True)
    ax.set_xlabel('Predicted Probability of Human Cause')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by True Class')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'experiment_results.png', dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_PATH / 'experiment_results.png'}")
except Exception as e:
    print(f"Visualization error: {e}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
