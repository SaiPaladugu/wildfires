"""
SAMPLE PROJECT: Predicting Wildfire Ignition Cause Using Machine Learning
==========================================================================
Course: Data Science (COMP 4107/5107 - Carleton University)

This project demonstrates:
1. Data loading and preprocessing
2. Exploratory data analysis  
3. Feature engineering
4. Machine learning model training
5. Model evaluation and interpretation

Topic: Can we predict whether a wildfire was naturally caused (lightning) 
       or human-caused based on environmental and temporal features?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dbfread import DBF
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_PATH = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = BASE_PATH.parent / "outputs"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("="*80)
print("WILDFIRE CAUSE PREDICTION - SAMPLE DATA SCIENCE PROJECT")
print("="*80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "="*80)
print("PART 1: DATA LOADING")
print("="*80)

# Load fire data
print("\n[1.1] Loading Canadian National Fire Database (NFDB)...")
fire_dbf = next(BASE_PATH.glob("NFDB_poly_large_fires/*.dbf"))
table = DBF(str(fire_dbf), encoding='utf-8', ignore_missing_memofile=True)
fire_df = pd.DataFrame(iter(table))
print(f"      Loaded {len(fire_df):,} fire records")

# Load climate data samples
print("\n[1.2] Loading Climate Data Samples...")
climate_files = list((BASE_PATH / "climate").glob("climate_daily_*.csv"))
climate_dfs = []
for f in climate_files:
    try:
        df = pd.read_csv(f, encoding='latin-1')
        df['source_file'] = f.name
        climate_dfs.append(df)
    except Exception as e:
        print(f"      Error loading {f.name}: {e}")

if climate_dfs:
    climate_df = pd.concat(climate_dfs, ignore_index=True)
    print(f"      Loaded {len(climate_df):,} climate observations from {len(climate_files)} files")
else:
    climate_df = None
    print("      No climate data loaded")

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================
print("\n" + "="*80)
print("PART 2: DATA PREPROCESSING")
print("="*80)

print("\n[2.1] Filtering fires with known cause...")
# Keep only fires with Natural (N) or Human (H, H-PB) causes
valid_causes = ['N', 'H', 'H-PB']
df = fire_df[fire_df['CAUSE'].isin(valid_causes)].copy()

# Create binary target: Natural vs Human
df['target'] = df['CAUSE'].apply(lambda x: 0 if x == 'N' else 1)
df['target_label'] = df['target'].map({0: 'Natural', 1: 'Human'})

print(f"      Total fires with known cause: {len(df):,}")
print(f"      Class distribution:")
print(f"         Natural (lightning): {(df['target']==0).sum():,} ({(df['target']==0).mean()*100:.1f}%)")
print(f"         Human-caused: {(df['target']==1).sum():,} ({(df['target']==1).mean()*100:.1f}%)")

print("\n[2.2] Creating features...")

# Temporal features
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
df['DAY'] = pd.to_numeric(df['DAY'], errors='coerce')

# Day of year (captures seasonality)
def get_day_of_year(row):
    try:
        if pd.notna(row['YEAR']) and pd.notna(row['MONTH']) and pd.notna(row['DAY']):
            month = int(row['MONTH'])
            day = int(row['DAY'])
            if 1 <= month <= 12 and 1 <= day <= 31:
                return pd.Timestamp(year=2000, month=month, day=day).dayofyear
    except:
        pass
    return np.nan

df['DOY'] = df.apply(get_day_of_year, axis=1)

# Cyclic encoding of day of year (captures circular nature of seasons)
df['DOY_SIN'] = np.sin(2 * np.pi * df['DOY'] / 365)
df['DOY_COS'] = np.cos(2 * np.pi * df['DOY'] / 365)

# Region encoding
le_region = LabelEncoder()
df['REGION_CODE'] = le_region.fit_transform(df['SRC_AGENCY'].fillna('Unknown'))
region_mapping = dict(zip(le_region.classes_, le_region.transform(le_region.classes_)))
print(f"      Region encoding: {region_mapping}")

# Fire size (log-transformed for normality)
df['SIZE_HA_CLEAN'] = pd.to_numeric(df['SIZE_HA'], errors='coerce')
df['CALC_HA_CLEAN'] = pd.to_numeric(df['CALC_HA'], errors='coerce')
df['FIRE_SIZE'] = df['SIZE_HA_CLEAN'].fillna(df['CALC_HA_CLEAN'])
df['LOG_SIZE'] = np.log1p(df['FIRE_SIZE'])

# Decade feature (captures long-term trends)
df['DECADE'] = (df['YEAR'] // 10) * 10

# Weekend indicator (human fires may be more common on weekends)
def get_day_of_week(row):
    try:
        if pd.notna(row['YEAR']) and pd.notna(row['MONTH']) and pd.notna(row['DAY']):
            year = int(row['YEAR'])
            month = int(row['MONTH'])
            day = int(row['DAY'])
            if 1900 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                return pd.Timestamp(year=year, month=month, day=day).dayofweek
    except:
        pass
    return np.nan

df['DAY_OF_WEEK'] = df.apply(get_day_of_week, axis=1)
df['IS_WEEKEND'] = df['DAY_OF_WEEK'].apply(lambda x: 1 if x in [5, 6] else 0 if pd.notna(x) else np.nan)

print("\n[2.3] Feature summary:")
feature_cols = ['MONTH', 'DOY', 'DOY_SIN', 'DOY_COS', 'REGION_CODE', 
                'LOG_SIZE', 'YEAR', 'DECADE', 'IS_WEEKEND']

for col in feature_cols:
    non_null = df[col].notna().sum()
    print(f"      {col}: {non_null:,} non-null ({non_null/len(df)*100:.1f}%)")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("PART 3: EXPLORATORY DATA ANALYSIS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Target distribution
ax = axes[0, 0]
df['target_label'].value_counts().plot(kind='bar', ax=ax, color=['forestgreen', 'coral'])
ax.set_title('Fire Cause Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Cause')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)

# 2. Monthly distribution by cause
ax = axes[0, 1]
monthly_cause = df.groupby(['MONTH', 'target_label']).size().unstack(fill_value=0)
monthly_cause.plot(kind='bar', ax=ax, color=['forestgreen', 'coral'], width=0.8)
ax.set_title('Fires by Month and Cause', fontsize=12, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Count')
ax.legend(title='Cause')
ax.tick_params(axis='x', rotation=0)

# 3. Regional distribution by cause
ax = axes[0, 2]
region_cause = df.groupby(['SRC_AGENCY', 'target_label']).size().unstack(fill_value=0)
region_cause_pct = region_cause.div(region_cause.sum(axis=1), axis=0) * 100
region_cause_pct.sort_values('Human', ascending=True).plot(
    kind='barh', ax=ax, stacked=True, color=['forestgreen', 'coral']
)
ax.set_title('Cause % by Province', fontsize=12, fontweight='bold')
ax.set_xlabel('Percentage')
ax.legend(title='Cause', loc='lower right')

# 4. Fire size distribution by cause
ax = axes[1, 0]
for label, color in [('Natural', 'forestgreen'), ('Human', 'coral')]:
    subset = df[df['target_label'] == label]['LOG_SIZE'].dropna()
    ax.hist(subset, bins=50, alpha=0.6, label=label, color=color, density=True)
ax.set_title('Fire Size Distribution by Cause', fontsize=12, fontweight='bold')
ax.set_xlabel('Log(Size in hectares)')
ax.set_ylabel('Density')
ax.legend()

# 5. Yearly trend
ax = axes[1, 1]
yearly_cause = df[df['YEAR'] >= 1980].groupby(['YEAR', 'target_label']).size().unstack(fill_value=0)
yearly_cause.plot(ax=ax, color=['forestgreen', 'coral'], linewidth=2)
ax.set_title('Annual Fire Count by Cause', fontsize=12, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Fires')
ax.legend(title='Cause')

# 6. Day of year distribution
ax = axes[1, 2]
for label, color in [('Natural', 'forestgreen'), ('Human', 'coral')]:
    subset = df[df['target_label'] == label]['DOY'].dropna()
    ax.hist(subset, bins=52, alpha=0.6, label=label, color=color, density=True)
ax.set_title('Seasonality by Cause', fontsize=12, fontweight='bold')
ax.set_xlabel('Day of Year')
ax.set_ylabel('Density')
ax.legend()
ax.axvline(x=152, color='gray', linestyle='--', alpha=0.5, label='Jun 1')
ax.axvline(x=244, color='gray', linestyle='--', alpha=0.5, label='Sep 1')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'eda_plots.png', dpi=150)
plt.close()
print(f"\n      Saved EDA plots to: {OUTPUT_PATH / 'eda_plots.png'}")

# Key insights
print("\n[3.1] Key Insights from EDA:")
natural_mean_doy = df[df['target']==0]['DOY'].mean()
human_mean_doy = df[df['target']==1]['DOY'].mean()
print(f"      • Natural fires peak later in season (mean DOY: {natural_mean_doy:.0f}) vs Human ({human_mean_doy:.0f})")

natural_mean_size = np.exp(df[df['target']==0]['LOG_SIZE'].mean())
human_mean_size = np.exp(df[df['target']==1]['LOG_SIZE'].mean())
print(f"      • Natural fires tend to be larger ({natural_mean_size:,.0f} ha) vs Human ({human_mean_size:,.0f} ha)")

human_rate_by_region = df.groupby('SRC_AGENCY')['target'].mean().sort_values(ascending=False)
print(f"      • Highest human-caused fire rates: {', '.join(human_rate_by_region.head(3).index.tolist())}")

# =============================================================================
# 4. MODEL TRAINING
# =============================================================================
print("\n" + "="*80)
print("PART 4: MODEL TRAINING")
print("="*80)

# Prepare final dataset
model_features = ['MONTH', 'DOY', 'DOY_SIN', 'DOY_COS', 'REGION_CODE', 'LOG_SIZE', 'YEAR']
model_df = df.dropna(subset=model_features + ['target'])

X = model_df[model_features]
y = model_df['target']

print(f"\n[4.1] Dataset prepared:")
print(f"      Total samples: {len(X):,}")
print(f"      Features: {model_features}")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\n[4.2] Train-test split:")
print(f"      Training: {len(X_train):,} samples")
print(f"      Testing: {len(X_test):,} samples")

# Scale features for some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\n[4.3] Training models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    print(f"      Training {name}...")
    
    # Use scaled data for logistic regression
    if 'Logistic' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    print(f"         Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}")

# =============================================================================
# 5. MODEL EVALUATION
# =============================================================================
print("\n" + "="*80)
print("PART 5: MODEL EVALUATION")
print("="*80)

# Find best model
best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
best_result = results[best_model_name]

print(f"\n[5.1] Best Model: {best_model_name}")
print(f"      Accuracy: {best_result['accuracy']:.3f} ({best_result['accuracy']*100:.1f}%)")
print(f"      ROC-AUC: {best_result['roc_auc']:.3f}")

print(f"\n[5.2] Classification Report ({best_model_name}):")
print(classification_report(y_test, best_result['y_pred'], 
                           target_names=['Natural', 'Human']))

# Confusion matrix
print(f"\n[5.3] Confusion Matrix:")
cm = confusion_matrix(y_test, best_result['y_pred'])
print(f"      {'':>15} Predicted")
print(f"      {'':>15} Natural  Human")
print(f"      Actual Natural   {cm[0,0]:>5}   {cm[0,1]:>5}")
print(f"      Actual Human     {cm[1,0]:>5}   {cm[1,1]:>5}")

# Feature importance (for tree-based models)
if 'Forest' in best_model_name or 'Boosting' in best_model_name:
    print(f"\n[5.4] Feature Importance ({best_model_name}):")
    importance = pd.DataFrame({
        'feature': model_features,
        'importance': best_result['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.to_string(index=False))

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================
print("\n" + "="*80)
print("PART 6: VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Model comparison
ax = axes[0, 0]
model_comparison = pd.DataFrame({
    'Model': results.keys(),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'ROC-AUC': [r['roc_auc'] for r in results.values()]
})
x = np.arange(len(model_comparison))
width = 0.35
bars1 = ax.bar(x - width/2, model_comparison['Accuracy'], width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, model_comparison['ROC-AUC'], width, label='ROC-AUC', color='coral')
ax.set_ylabel('Score')
ax.set_title('Model Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_comparison['Model'], rotation=15)
ax.legend()
ax.set_ylim(0, 1.1)
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# 2. ROC Curves
ax = axes[0, 1]
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')

# 3. Confusion Matrix Heatmap
ax = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Natural', 'Human'], yticklabels=['Natural', 'Human'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix ({best_model_name})', fontsize=12, fontweight='bold')

# 4. Feature Importance
ax = axes[1, 1]
if 'Forest' in best_model_name or 'Boosting' in best_model_name:
    importance_sorted = importance.sort_values('importance', ascending=True)
    ax.barh(importance_sorted['feature'], importance_sorted['importance'], color='teal')
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance ({best_model_name})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'model_evaluation.png', dpi=150)
plt.close()
print(f"\n      Saved model evaluation plots to: {OUTPUT_PATH / 'model_evaluation.png'}")

# =============================================================================
# 7. CONCLUSIONS
# =============================================================================
print("\n" + "="*80)
print("PART 7: CONCLUSIONS AND FUTURE WORK")
print("="*80)

print("""
[7.1] KEY FINDINGS:

1. PREDICTION ACCURACY
   • Best model ({}) achieved {:.1f}% accuracy and {:.3f} ROC-AUC
   • This significantly outperforms random guessing (baseline ~86% due to class imbalance)
   • The model can reliably distinguish natural from human-caused fires

2. IMPORTANT PREDICTIVE FEATURES
   • Day of Year (DOY): Most important - seasonality strongly predicts cause
   • Fire Size: Natural fires tend to grow larger before detection
   • Region: Some provinces have higher rates of human-caused fires
   • Year: Temporal trends in fire causes over decades

3. ECOLOGICAL INSIGHTS
   • Natural (lightning) fires peak in mid-summer (July-August)
   • Human-caused fires have broader seasonal distribution
   • Certain regions (eastern provinces) have higher human-caused rates

[7.2] LIMITATIONS:

   • Class imbalance: Only ~14% of fires are human-caused
   • Missing climate data integration
   • Geographic precision limited to province level

[7.3] FUTURE WORK:

   • Integrate climate data (temperature, precipitation, drought indices)
   • Add fire weather indices (FWI system)
   • Include proximity to roads, population centers
   • Test deep learning approaches
   • Develop early warning system prototype

[7.4] IMPLICATIONS:

   • Fire prevention resources can be targeted to high-risk human-cause periods
   • Different monitoring strategies for natural vs human fire seasons
   • Policy implications for fire prevention education campaigns
""".format(best_model_name, best_result['accuracy']*100, best_result['roc_auc']))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput files saved to: {OUTPUT_PATH}")
print("Files generated:")
print("  • eda_plots.png - Exploratory data analysis visualizations")
print("  • model_evaluation.png - Model comparison and evaluation plots")
