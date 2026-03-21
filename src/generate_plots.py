"""
Generate publication-quality plots for the wildfire cause prediction report.
Loads cached data, re-trains the GBDT (deterministic), produces all figures.
"""

import urllib.request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
import shap
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

CANADA_XLIM = (-145, -50)
CANADA_YLIM = (42, 85)

# Natural Earth URLs — 50m resolution is a good balance of detail vs. file size
_NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
_BOUNDARY_CACHE = BASE_PATH / "boundaries"

def _load_boundaries():
    """Return (provinces_gdf, countries_gdf), downloading and caching on first run."""
    _BOUNDARY_CACHE.mkdir(exist_ok=True)
    provinces_path = _BOUNDARY_CACHE / "ne_50m_admin_1_provinces.geojson"
    countries_path = _BOUNDARY_CACHE / "ne_50m_admin_0_countries.geojson"

    if not provinces_path.exists():
        print("  Downloading provincial boundaries (one-time)...")
        urllib.request.urlretrieve(
            f"{_NE_BASE}/ne_50m_admin_1_states_provinces.geojson", provinces_path
        )
    if not countries_path.exists():
        print("  Downloading country boundaries (one-time)...")
        urllib.request.urlretrieve(
            f"{_NE_BASE}/ne_50m_admin_0_countries.geojson", countries_path
        )

    provinces = gpd.read_file(provinces_path)
    countries = gpd.read_file(countries_path)
    return provinces[provinces['iso_a2'] == 'CA'], countries[countries['ISO_A2'] == 'CA']


def add_canada_map(ax, provinces, canada):
    """Draw Canada outline and provincial boundaries on ax."""
    canada.plot(ax=ax, facecolor='#f5f5f0', edgecolor='#555555', linewidth=1.2, zorder=0)
    provinces.plot(ax=ax, facecolor='none', edgecolor='#888888', linewidth=0.5, zorder=1)
    ax.set_xlim(CANADA_XLIM)
    ax.set_ylim(CANADA_YLIM)
    ax.set_aspect('equal')


plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# =============================================================================
# LOAD DATA AND RE-TRAIN MODEL (deterministic)
# =============================================================================
print("Loading cached data...")
df = pd.read_parquet(BASE_PATH / "joined_fire_climate.parquet")
print(f"Loaded {len(df):,} records")

# Feature engineering (replicate from experiment.py)
def safe_doy(row):
    try:
        return pd.Timestamp(year=2000, month=int(row['MONTH']), day=int(row['DAY'])).dayofyear
    except:
        return np.nan

df['doy'] = df.apply(safe_doy, axis=1)
df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)

def safe_dow(row):
    try:
        return pd.Timestamp(year=int(row['YEAR']), month=int(row['MONTH']), day=int(row['DAY'])).dayofweek
    except:
        return np.nan

df['dow'] = df.apply(safe_dow, axis=1)
df['is_weekend'] = (df['dow'] >= 5).astype(float)
df['month'] = pd.to_numeric(df['MONTH'], errors='coerce')
df['year'] = pd.to_numeric(df['YEAR'], errors='coerce')

province_map = {v: i for i, v in enumerate(sorted(df['SRC_AGENCY'].dropna().unique()))}
df['province_code'] = df['SRC_AGENCY'].map(province_map)
df['latitude'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['longitude'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df['log_size'] = np.log1p(pd.to_numeric(df['SIZE_HA'], errors='coerce').clip(lower=0))
df['target'] = df['CAUSE'].apply(lambda x: 0 if x == 'N' else 1)

if 'temp_max' in df.columns and 'temp_min' in df.columns:
    df['diurnal_range'] = pd.to_numeric(df.get('temp_max'), errors='coerce') - pd.to_numeric(df.get('temp_min'), errors='coerce')
if 'temp_mean' in df.columns and 'temp_mean_30d' in df.columns:
    df['temp_anomaly'] = pd.to_numeric(df.get('temp_mean'), errors='coerce') - pd.to_numeric(df.get('temp_mean_30d'), errors='coerce')

for col in ['temp_max', 'temp_min', 'temp_mean', 'precip', 'rain', 'snow_ground', 'wind_gust',
            'temp_mean_7d', 'temp_max_7d', 'precip_total_7d', 'rain_total_7d', 'dry_days_7d', 'temp_range_7d',
            'temp_mean_30d', 'temp_max_30d', 'precip_total_30d', 'rain_total_30d', 'dry_days_30d', 'temp_range_30d']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if all(c in df.columns for c in ['temp_max', 'precip', 'wind_gust']):
    df['fire_weather_index'] = df['temp_max'].fillna(0) - df['precip'].fillna(0) * 2 + df['wind_gust'].fillna(0) * 0.5

if 'station_dist_km' in df.columns:
    df['station_dist'] = df['station_dist_km']

feature_cols = [
    'doy', 'doy_sin', 'doy_cos', 'month', 'year', 'is_weekend',
    'province_code', 'latitude', 'longitude', 'log_size',
    'temp_max', 'temp_min', 'temp_mean', 'precip', 'rain',
    'snow_ground', 'wind_gust', 'diurnal_range',
    'temp_mean_7d', 'temp_max_7d', 'precip_total_7d', 'rain_total_7d',
    'dry_days_7d', 'temp_range_7d',
    'temp_mean_30d', 'temp_max_30d', 'precip_total_30d', 'rain_total_30d',
    'dry_days_30d', 'temp_range_30d',
    'temp_anomaly', 'fire_weather_index', 'station_dist',
]
feature_cols = [c for c in feature_cols if c in df.columns]

core_features = ['doy', 'province_code', 'latitude', 'longitude', 'log_size', 'year']
model_df = df.dropna(subset=core_features + ['target']).copy()

X = model_df[feature_cols].fillna(-999)
y = model_df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

n_neg, n_pos = sum(y_train == 0), sum(y_train == 1)
sample_weights = np.where(y_train == 1, n_neg / n_pos, 1.0)

print("Loading boundary data...")
canada_provinces, canada_outline = _load_boundaries()

print("Training GBDT...")
gbdt = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=5, min_samples_leaf=20,
    subsample=0.8, max_features='sqrt', random_state=RANDOM_STATE,
    validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
)
gbdt.fit(X_train, y_train, sample_weight=sample_weights)
print(f"Trained ({gbdt.n_estimators_} trees)")

FEATURE_LABELS = {
    'doy_sin': 'Day of Year (sin)', 'doy_cos': 'Day of Year (cos)', 'doy': 'Day of Year',
    'month': 'Month', 'year': 'Year', 'is_weekend': 'Weekend',
    'province_code': 'Province', 'latitude': 'Latitude', 'longitude': 'Longitude',
    'log_size': 'Fire Size (log ha)', 'temp_max': 'Max Temp (°C)',
    'temp_min': 'Min Temp (°C)', 'temp_mean': 'Mean Temp (°C)',
    'precip': 'Precipitation (mm)', 'rain': 'Rainfall (mm)',
    'snow_ground': 'Snow on Ground (cm)', 'wind_gust': 'Wind Gust (km/h)',
    'diurnal_range': 'Diurnal Temp Range (°C)',
    'temp_mean_7d': '7-day Mean Temp', 'temp_max_7d': '7-day Max Temp',
    'precip_total_7d': '7-day Precip Total', 'rain_total_7d': '7-day Rain Total',
    'dry_days_7d': '7-day Dry Days', 'temp_range_7d': '7-day Temp Range',
    'temp_mean_30d': '30-day Mean Temp', 'temp_max_30d': '30-day Max Temp',
    'precip_total_30d': '30-day Precip Total', 'rain_total_30d': '30-day Rain Total',
    'dry_days_30d': '30-day Dry Days', 'temp_range_30d': '30-day Temp Range',
    'temp_anomaly': 'Temp Anomaly (°C)', 'fire_weather_index': 'Fire Weather Index',
    'station_dist': 'Station Distance (km)',
}

# =============================================================================
# PLOT 1: SHAP Beeswarm
# =============================================================================
print("Generating SHAP beeswarm...")
rng = np.random.RandomState(RANDOM_STATE)
shap_idx = rng.choice(len(X_test), size=min(5000, len(X_test)), replace=False)
X_shap = X_test.iloc[shap_idx]

explainer = shap.TreeExplainer(gbdt)
shap_values = explainer.shap_values(X_shap)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    shap_values, X_shap,
    feature_names=[FEATURE_LABELS.get(c, c) for c in feature_cols],
    show=False, max_display=20
)
plt.title('SHAP Feature Importance (Beeswarm)', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'shap_beeswarm.png')
plt.close()
print(f"  Saved shap_beeswarm.png")

# =============================================================================
# PLOT 2: Gain-based Feature Importance (top 20)
# =============================================================================
print("Generating feature importance (gain)...")
importance = pd.DataFrame({
    'feature': [FEATURE_LABELS.get(c, c) for c in feature_cols],
    'importance': gbdt.feature_importances_
}).sort_values('importance', ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(8, 7))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance)))
ax.barh(importance['feature'], importance['importance'], color=colors)
ax.set_xlabel('Mean Decrease in Impurity (Gini Importance)')
ax.set_title('Top 20 Features — Gain-Based Importance', fontsize=13, fontweight='bold')
for i, (_, row) in enumerate(importance.iterrows()):
    ax.text(row['importance'] + 0.002, i, f"{row['importance']:.3f}", va='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'feature_importance_gain.png')
plt.close()
print(f"  Saved feature_importance_gain.png")

# =============================================================================
# PLOT 3: Partial Dependence Plots
# =============================================================================
print("Generating partial dependence plots...")
pdp_features = ['doy_sin', 'latitude', 'longitude', 'month']
pdp_indices = [feature_cols.index(f) for f in pdp_features if f in feature_cols]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
display = PartialDependenceDisplay.from_estimator(
    gbdt, X_train.sample(10000, random_state=RANDOM_STATE),
    features=pdp_indices, feature_names=[FEATURE_LABELS.get(c, c) for c in feature_cols],
    ax=axes.ravel(), kind='average', grid_resolution=50,
)
fig.suptitle('Partial Dependence Plots — Effect on P(Human Cause)', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PATH / 'partial_dependence.png')
plt.close()
print(f"  Saved partial_dependence.png")

# =============================================================================
# PLOT 4: Class Distribution by Key Features
# =============================================================================
print("Generating class distribution plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) By month
ax = axes[0, 0]
monthly = model_df.groupby(['month', 'target']).size().unstack(fill_value=0)
monthly.columns = ['Natural', 'Human']
monthly.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], width=0.8)
ax.set_title('Fire Cause by Month', fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Count')
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], rotation=0)
ax.legend()

# (b) By province
ax = axes[0, 1]
prov_cause = model_df.groupby(['SRC_AGENCY', 'target']).size().unstack(fill_value=0)
prov_cause.columns = ['Natural', 'Human']
prov_pct = prov_cause.div(prov_cause.sum(axis=1), axis=0) * 100
prov_pct.sort_values('Human', ascending=True).plot(
    kind='barh', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c']
)
ax.set_title('Cause Proportion by Province', fontweight='bold')
ax.set_xlabel('Percentage (%)')
ax.legend(loc='lower right')

# (c) By decade
ax = axes[1, 0]
model_df['decade'] = (model_df['year'] // 10) * 10
decade_cause = model_df.groupby(['decade', 'target']).size().unstack(fill_value=0)
decade_cause.columns = ['Natural', 'Human']
decade_cause.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], width=0.8)
ax.set_title('Fire Cause by Decade', fontweight='bold')
ax.set_xlabel('Decade')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
ax.legend()

# (d) By day of week
ax = axes[1, 1]
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_cause = model_df.groupby(['dow', 'target']).size().unstack(fill_value=0)
dow_cause.columns = ['Natural', 'Human']
dow_pct = dow_cause.div(dow_cause.sum(axis=1), axis=0) * 100
dow_pct.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], width=0.8)
ax.set_title('Cause Proportion by Day of Week', fontweight='bold')
ax.set_xlabel('Day')
ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(dow_names, rotation=0)
ax.legend()

plt.suptitle('Class Distribution Analysis', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_PATH / 'class_distribution.png')
plt.close()
print(f"  Saved class_distribution.png")

# =============================================================================
# PLOT 5: Geographic Scatter
# =============================================================================
print("Generating geographic scatter...")
sample = model_df.sample(min(50000, len(model_df)), random_state=RANDOM_STATE)

fig, ax = plt.subplots(figsize=(12, 8))
add_canada_map(ax, canada_provinces, canada_outline)
natural = sample[sample['target'] == 0]
human = sample[sample['target'] == 1]
ax.scatter(natural['longitude'], natural['latitude'], c='#2ecc71', alpha=0.08, s=3, label='Natural', rasterized=True)
ax.scatter(human['longitude'], human['latitude'], c='#e74c3c', alpha=0.08, s=3, label='Human', rasterized=True)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Geographic Distribution of Fires by Cause', fontsize=13, fontweight='bold')
ax.legend(markerscale=5, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'geographic_scatter.png')
plt.close()
print(f"  Saved geographic_scatter.png")

# =============================================================================
# PLOT 6: Temporal Trend
# =============================================================================
print("Generating temporal trend...")
yearly = model_df[(model_df['year'] >= 1972) & (model_df['year'] <= 2024)]
yearly_cause = yearly.groupby(['year', 'target']).size().unstack(fill_value=0)
yearly_cause.columns = ['Natural', 'Human']

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(yearly_cause.index, yearly_cause['Natural'], alpha=0.4, color='#2ecc71', label='Natural')
ax.fill_between(yearly_cause.index, yearly_cause['Human'], alpha=0.4, color='#e74c3c', label='Human')
ax.plot(yearly_cause.index, yearly_cause['Natural'], color='#27ae60', lw=1.5)
ax.plot(yearly_cause.index, yearly_cause['Human'], color='#c0392b', lw=1.5)
ax.set_xlabel('Year')
ax.set_ylabel('Number of Fires')
ax.set_title('Annual Fire Count by Cause (1972–2024)', fontsize=13, fontweight='bold')
ax.legend()
ax.set_xlim(1972, 2024)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'temporal_trend.png')
plt.close()
print(f"  Saved temporal_trend.png")

# =============================================================================
# PLOT 7: Climate Feature Correlation Heatmap
# =============================================================================
print("Generating correlation heatmap...")
climate_feats = ['temp_max', 'temp_min', 'temp_mean', 'precip', 'rain',
                 'diurnal_range', 'temp_mean_7d', 'precip_total_7d',
                 'dry_days_7d', 'temp_mean_30d', 'precip_total_30d',
                 'dry_days_30d', 'temp_anomaly', 'fire_weather_index']
climate_feats = [c for c in climate_feats if c in model_df.columns]

corr_data = model_df[climate_feats].replace(-999, np.nan).dropna()
corr = corr_data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
labels = [FEATURE_LABELS.get(c, c) for c in climate_feats]
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=labels, yticklabels=labels, ax=ax,
            square=True, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Climate Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'climate_correlation.png')
plt.close()
print(f"  Saved climate_correlation.png")

# =============================================================================
# PLOT 8: Weather Station Locations
# =============================================================================
print("Generating station locations map...")
stations_csv = BASE_PATH / "stations_1972_2024.csv"
source_csv = BASE_PATH / "climate-stations.csv"

stations_filtered = pd.read_csv(stations_csv)
source = pd.read_csv(source_csv)
stations_geo = stations_filtered.merge(
    source[['STN_ID', 'x', 'y']], left_on='STATION_ID', right_on='STN_ID', how='left'
).dropna(subset=['x', 'y'])

fig, ax = plt.subplots(figsize=(12, 8))
add_canada_map(ax, canada_provinces, canada_outline)
ax.scatter(
    stations_geo['x'], stations_geo['y'],
    c='#e67e22', s=25, zorder=5, edgecolors='#333333', linewidths=0.4,
    label=f'Weather stations (n={len(stations_geo)})'
)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Environment Canada Weather Station Locations (1972–2024 coverage)',
             fontsize=13, fontweight='bold')
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'station_locations.png')
plt.close()
print(f"  Saved station_locations.png")

print("\nAll plots generated successfully!")
print(f"Output directory: {OUTPUT_PATH}")
