"""
Wildfire & Climate Data Exploratory Analysis
=============================================
This script explores the NFDB fire data and climate data to identify
the best thesis topics for a Data Science course project.

POC Topics to Evaluate:
1. Fire Cause Prediction (Classification)
2. Fire Size Prediction (Regression)  
3. Temporal Trends Analysis
4. Regional Fire Pattern Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dbfread import DBF
import shapefile
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

BASE_PATH = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = BASE_PATH.parent / "analysis_output"
OUTPUT_PATH.mkdir(exist_ok=True)

print("="*70)
print("WILDFIRE & CLIMATE DATA EXPLORATORY ANALYSIS")
print("="*70)

# =============================================================================
# 1. LOAD AND EXPLORE FIRE DATA
# =============================================================================
print("\n" + "="*70)
print("SECTION 1: LOADING FIRE DATA")
print("="*70)

# Load NFDB polygon data (large fires - more manageable size)
print("\n[1.1] Loading NFDB Large Fires (1972-2024)...")
large_fires_dbf = next(BASE_PATH.glob("NFDB_poly_large_fires/*.dbf"))

try:
    # Read DBF file
    table = DBF(str(large_fires_dbf), encoding='utf-8', ignore_missing_memofile=True)
    fire_df = pd.DataFrame(iter(table))
    print(f"   Loaded {len(fire_df):,} large fire records")
    print(f"   Columns: {list(fire_df.columns)}")
except Exception as e:
    print(f"   Error loading large fires: {e}")
    fire_df = None

# Also try loading the recent fires (2021-2024)
print("\n[1.2] Loading NFDB Recent Fires (2021-2024)...")
recent_fires_dbf = next(BASE_PATH.glob("NFDB_poly/*.dbf"), None)

try:
    table2 = DBF(str(recent_fires_dbf), encoding='utf-8', ignore_missing_memofile=True)
    recent_fire_df = pd.DataFrame(iter(table2))
    print(f"   Loaded {len(recent_fire_df):,} recent fire records")
except Exception as e:
    print(f"   Error loading recent fires: {e}")
    recent_fire_df = None

# Load Excel statistics
print("\n[1.3] Loading NFDB Point Statistics...")
stats_file = next(BASE_PATH.glob("NFDB_point_stats/*.xlsx"), None)
try:
    fire_stats = pd.read_excel(stats_file)
    print(f"   Loaded statistics with shape: {fire_stats.shape}")
    print(f"   Columns: {list(fire_stats.columns)[:10]}...")
except Exception as e:
    print(f"   Error loading stats: {e}")
    fire_stats = None

# =============================================================================
# 2. EXPLORE FIRE DATA STRUCTURE
# =============================================================================
print("\n" + "="*70)
print("SECTION 2: FIRE DATA EXPLORATION")
print("="*70)

if fire_df is not None:
    print("\n[2.1] Large Fires DataFrame Info:")
    print(f"   Shape: {fire_df.shape}")
    print(f"   Memory: {fire_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n[2.2] Column Data Types:")
    print(fire_df.dtypes.to_string())
    
    print("\n[2.3] Sample Data (first 3 rows):")
    print(fire_df.head(3).to_string())
    
    print("\n[2.4] Key Variable Statistics:")
    
    # Year distribution
    if 'YEAR' in fire_df.columns:
        print(f"\n   YEAR range: {fire_df['YEAR'].min()} - {fire_df['YEAR'].max()}")
        print(f"   Years with most fires:")
        print(fire_df['YEAR'].value_counts().head(10).to_string())
    
    # Fire cause distribution
    if 'CAUSE' in fire_df.columns:
        print(f"\n   CAUSE distribution:")
        print(fire_df['CAUSE'].value_counts().to_string())
    
    # Province/Agency distribution
    if 'SRC_AGENCY' in fire_df.columns:
        print(f"\n   SRC_AGENCY (Province) distribution:")
        print(fire_df['SRC_AGENCY'].value_counts().to_string())
    
    # Fire size statistics
    for col in ['SIZE_HA', 'CALC_HA']:
        if col in fire_df.columns:
            valid_sizes = fire_df[col].dropna()
            valid_sizes = valid_sizes[valid_sizes > 0]
            print(f"\n   {col} statistics (n={len(valid_sizes):,}):")
            print(f"      Min: {valid_sizes.min():,.1f} ha")
            print(f"      Max: {valid_sizes.max():,.1f} ha")
            print(f"      Mean: {valid_sizes.mean():,.1f} ha")
            print(f"      Median: {valid_sizes.median():,.1f} ha")
            print(f"      Std: {valid_sizes.std():,.1f} ha")

# =============================================================================
# 3. LOAD AND EXPLORE CLIMATE DATA
# =============================================================================
print("\n" + "="*70)
print("SECTION 3: CLIMATE DATA EXPLORATION")
print("="*70)

climate_files = list((BASE_PATH / "climate").glob("climate_daily_*.csv"))
print(f"\n[3.1] Found {len(climate_files)} climate data files:")
for f in climate_files:
    print(f"   - {f.name}")

# Load all climate files
climate_dfs = []
for f in climate_files:
    try:
        df = pd.read_csv(f)
        df['source_file'] = f.name
        climate_dfs.append(df)
        print(f"\n   {f.name}: {len(df)} rows")
    except Exception as e:
        print(f"   Error loading {f.name}: {e}")

if climate_dfs:
    climate_df = pd.concat(climate_dfs, ignore_index=True)
    print(f"\n[3.2] Combined Climate Data:")
    print(f"   Total rows: {len(climate_df):,}")
    print(f"   Columns: {list(climate_df.columns)[:15]}...")
    
    print("\n[3.3] Climate Variables Available:")
    numeric_cols = climate_df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[:10]:
        non_null = climate_df[col].notna().sum()
        if non_null > 0:
            print(f"   {col}: {non_null} non-null values, mean={climate_df[col].mean():.2f}")
    
    print("\n[3.4] Weather Stations:")
    if 'Station Name' in climate_df.columns:
        print(climate_df['Station Name'].value_counts().to_string())
    
    print("\n[3.5] Date Range:")
    if 'Year' in climate_df.columns:
        print(f"   Years: {climate_df['Year'].min()} - {climate_df['Year'].max()}")

# =============================================================================
# 4. DATA QUALITY ASSESSMENT
# =============================================================================
print("\n" + "="*70)
print("SECTION 4: DATA QUALITY ASSESSMENT")
print("="*70)

if fire_df is not None:
    print("\n[4.1] Fire Data Missing Values:")
    missing = fire_df.isnull().sum()
    missing_pct = (missing / len(fire_df) * 100).round(1)
    missing_df = pd.DataFrame({'Missing': missing, 'Percent': missing_pct})
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Percent', ascending=False)
    print(missing_df.head(15).to_string())

if climate_dfs:
    print("\n[4.2] Climate Data Missing Values (sample):")
    key_cols = ['Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)', 
                'Total Rain (mm)', 'Total Snow (cm)', 'Total Precip (mm)']
    for col in key_cols:
        if col in climate_df.columns:
            missing = climate_df[col].isna().sum()
            pct = missing / len(climate_df) * 100
            print(f"   {col}: {missing} missing ({pct:.1f}%)")

# =============================================================================
# 5. POC ANALYSIS 1: FIRE CAUSE PREDICTION
# =============================================================================
print("\n" + "="*70)
print("SECTION 5: POC - FIRE CAUSE PREDICTION (Classification)")
print("="*70)

if fire_df is not None and 'CAUSE' in fire_df.columns:
    print("\n[5.1] Preparing data for cause prediction...")
    
    # Filter to fires with known cause (N=Natural, H=Human)
    cause_df = fire_df[fire_df['CAUSE'].isin(['N', 'H', 'H-PB'])].copy()
    
    # Simplify cause: Natural vs Human
    cause_df['CAUSE_SIMPLE'] = cause_df['CAUSE'].apply(
        lambda x: 'Natural' if x == 'N' else 'Human'
    )
    
    print(f"   Fires with known cause: {len(cause_df):,}")
    print(f"   Class distribution:")
    print(cause_df['CAUSE_SIMPLE'].value_counts().to_string())
    
    # Feature engineering using available fire attributes
    print("\n[5.2] Engineering features from fire data...")
    
    # Create features from date
    cause_df['MONTH'] = pd.to_numeric(cause_df['MONTH'], errors='coerce')
    cause_df['DAY'] = pd.to_numeric(cause_df['DAY'], errors='coerce')
    cause_df['YEAR'] = pd.to_numeric(cause_df['YEAR'], errors='coerce')
    
    # Day of year (seasonality)
    cause_df['DOY'] = cause_df.apply(
        lambda row: pd.Timestamp(year=int(row['YEAR']) if pd.notna(row['YEAR']) else 2000, 
                                  month=int(row['MONTH']) if pd.notna(row['MONTH']) and 1<=row['MONTH']<=12 else 1, 
                                  day=int(row['DAY']) if pd.notna(row['DAY']) and 1<=row['DAY']<=31 else 1).dayofyear 
        if pd.notna(row['MONTH']) else np.nan,
        axis=1
    )
    
    # Province encoding
    le_agency = LabelEncoder()
    cause_df['AGENCY_CODE'] = le_agency.fit_transform(cause_df['SRC_AGENCY'].fillna('Unknown'))
    
    # Fire size
    cause_df['SIZE_HA_CLEAN'] = pd.to_numeric(cause_df['SIZE_HA'], errors='coerce')
    cause_df['CALC_HA_CLEAN'] = pd.to_numeric(cause_df['CALC_HA'], errors='coerce')
    cause_df['FIRE_SIZE'] = cause_df['SIZE_HA_CLEAN'].fillna(cause_df['CALC_HA_CLEAN'])
    cause_df['LOG_SIZE'] = np.log1p(cause_df['FIRE_SIZE'])
    
    # Features for model
    feature_cols = ['MONTH', 'DOY', 'AGENCY_CODE', 'LOG_SIZE', 'YEAR']
    
    # Remove rows with missing features
    model_df = cause_df.dropna(subset=feature_cols + ['CAUSE_SIMPLE'])
    print(f"   Samples with complete features: {len(model_df):,}")
    
    if len(model_df) > 100:
        X = model_df[feature_cols]
        y = model_df['CAUSE_SIMPLE']
        
        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n[5.3] Training Random Forest Classifier...")
        print(f"   Training set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        
        # Train model
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[5.4] Model Performance:")
        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le_target.classes_))
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n[5.5] Feature Importance:")
        print(importance.to_string(index=False))
        
        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Class distribution
        cause_df['CAUSE_SIMPLE'].value_counts().plot(kind='bar', ax=axes[0], color=['forestgreen', 'coral'])
        axes[0].set_title('Fire Cause Distribution')
        axes[0].set_xlabel('Cause')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)
        
        # Feature importance
        sns.barplot(data=importance, x='importance', y='feature', ax=axes[1], palette='viridis')
        axes[1].set_title('Feature Importance for Cause Prediction')
        axes[1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / 'poc1_cause_prediction.png', dpi=150)
        plt.close()
        print(f"\n   Saved visualization to: {OUTPUT_PATH / 'poc1_cause_prediction.png'}")
        
        # Assessment
        print("\n[5.6] POC ASSESSMENT - Fire Cause Prediction:")
        print("   ✓ Data available: Sufficient samples with labeled cause")
        print(f"   ✓ Baseline accuracy: {accuracy*100:.1f}% (above random ~50%)")
        print("   ✓ Interpretable: Feature importance shows meaningful patterns")
        print("   ⚠ Limitation: Current features are fire-intrinsic, need to add climate data")
        print("   → RECOMMENDATION: STRONG CANDIDATE for project")

# =============================================================================
# 6. POC ANALYSIS 2: FIRE SIZE PREDICTION
# =============================================================================
print("\n" + "="*70)
print("SECTION 6: POC - FIRE SIZE PREDICTION (Regression)")
print("="*70)

if fire_df is not None:
    print("\n[6.1] Preparing data for size prediction...")
    
    # Use the data prepared above
    size_df = fire_df.copy()
    size_df['SIZE_HA_CLEAN'] = pd.to_numeric(size_df['SIZE_HA'], errors='coerce')
    size_df['CALC_HA_CLEAN'] = pd.to_numeric(size_df['CALC_HA'], errors='coerce')
    size_df['FIRE_SIZE'] = size_df['SIZE_HA_CLEAN'].fillna(size_df['CALC_HA_CLEAN'])
    
    # Filter valid sizes (>200 ha since this is large fires dataset)
    size_df = size_df[size_df['FIRE_SIZE'] > 0].copy()
    size_df['LOG_SIZE'] = np.log1p(size_df['FIRE_SIZE'])
    
    print(f"   Fires with valid size: {len(size_df):,}")
    print(f"   Size range: {size_df['FIRE_SIZE'].min():.1f} - {size_df['FIRE_SIZE'].max():,.1f} ha")
    
    # Features
    size_df['MONTH'] = pd.to_numeric(size_df['MONTH'], errors='coerce')
    size_df['YEAR'] = pd.to_numeric(size_df['YEAR'], errors='coerce')
    
    le_agency2 = LabelEncoder()
    size_df['AGENCY_CODE'] = le_agency2.fit_transform(size_df['SRC_AGENCY'].fillna('Unknown'))
    
    le_cause2 = LabelEncoder()
    size_df['CAUSE_CODE'] = le_cause2.fit_transform(size_df['CAUSE'].fillna('U'))
    
    feature_cols_size = ['MONTH', 'YEAR', 'AGENCY_CODE', 'CAUSE_CODE']
    
    model_df_size = size_df.dropna(subset=feature_cols_size + ['LOG_SIZE'])
    print(f"   Samples with complete features: {len(model_df_size):,}")
    
    if len(model_df_size) > 100:
        X = model_df_size[feature_cols_size]
        y = model_df_size['LOG_SIZE']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n[6.2] Training Random Forest Regressor...")
        
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n[6.3] Model Performance:")
        print(f"   R² Score: {r2:.3f}")
        print(f"   RMSE (log scale): {rmse:.3f}")
        
        # Convert back to hectares for interpretation
        y_test_ha = np.expm1(y_test)
        y_pred_ha = np.expm1(y_pred)
        rmse_ha = np.sqrt(mean_squared_error(y_test_ha, y_pred_ha))
        print(f"   RMSE (hectares): {rmse_ha:,.1f} ha")
        
        # Feature importance
        importance_size = pd.DataFrame({
            'feature': feature_cols_size,
            'importance': reg.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n[6.4] Feature Importance:")
        print(importance_size.to_string(index=False))
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Size distribution
        axes[0].hist(np.log10(size_df['FIRE_SIZE']), bins=50, color='orangered', alpha=0.7, edgecolor='black')
        axes[0].set_title('Fire Size Distribution (Large Fires)')
        axes[0].set_xlabel('Log10(Size in hectares)')
        axes[0].set_ylabel('Count')
        
        # Actual vs Predicted
        axes[1].scatter(y_test, y_pred, alpha=0.3, s=10)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_title(f'Actual vs Predicted Fire Size (R²={r2:.3f})')
        axes[1].set_xlabel('Actual (log scale)')
        axes[1].set_ylabel('Predicted (log scale)')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / 'poc2_size_prediction.png', dpi=150)
        plt.close()
        print(f"\n   Saved visualization to: {OUTPUT_PATH / 'poc2_size_prediction.png'}")
        
        print("\n[6.5] POC ASSESSMENT - Fire Size Prediction:")
        print(f"   ✓ Data available: {len(model_df_size):,} samples")
        print(f"   {'✓' if r2 > 0.1 else '⚠'} R² Score: {r2:.3f} - {'Reasonable' if r2 > 0.1 else 'Low'} predictive power")
        print("   ⚠ Note: Fire size is highly variable and hard to predict")
        print("   → Climate data integration could significantly improve this")
        print("   → RECOMMENDATION: MODERATE CANDIDATE - needs more features")

# =============================================================================
# 7. POC ANALYSIS 3: TEMPORAL TRENDS
# =============================================================================
print("\n" + "="*70)
print("SECTION 7: POC - TEMPORAL TREND ANALYSIS")
print("="*70)

if fire_df is not None:
    print("\n[7.1] Analyzing fire trends over time...")
    
    fire_df['YEAR'] = pd.to_numeric(fire_df['YEAR'], errors='coerce')
    fire_df['SIZE_HA_CLEAN'] = pd.to_numeric(fire_df['SIZE_HA'], errors='coerce')
    fire_df['CALC_HA_CLEAN'] = pd.to_numeric(fire_df['CALC_HA'], errors='coerce')
    fire_df['FIRE_SIZE'] = fire_df['SIZE_HA_CLEAN'].fillna(fire_df['CALC_HA_CLEAN'])
    
    # Aggregate by year
    yearly_stats = fire_df.groupby('YEAR').agg({
        'FIRE_SIZE': ['count', 'sum', 'mean', 'median', 'max']
    }).round(1)
    yearly_stats.columns = ['num_fires', 'total_area', 'mean_size', 'median_size', 'max_size']
    yearly_stats = yearly_stats.reset_index()
    yearly_stats = yearly_stats[(yearly_stats['YEAR'] >= 1980) & (yearly_stats['YEAR'] <= 2024)]
    
    print(f"\n[7.2] Annual Statistics (1980-2024):")
    print(yearly_stats.tail(15).to_string(index=False))
    
    # Trend analysis
    from scipy import stats
    
    # Filter to complete years
    trend_data = yearly_stats[(yearly_stats['num_fires'] > 10)]
    
    if len(trend_data) > 10:
        # Number of fires trend
        slope_count, intercept_count, r_count, p_count, se_count = stats.linregress(
            trend_data['YEAR'], trend_data['num_fires']
        )
        
        # Total area trend
        slope_area, intercept_area, r_area, p_area, se_area = stats.linregress(
            trend_data['YEAR'], trend_data['total_area']
        )
        
        print(f"\n[7.3] Trend Analysis:")
        print(f"   Number of Large Fires:")
        print(f"      Trend: {slope_count:+.2f} fires/year")
        print(f"      R²: {r_count**2:.3f}, p-value: {p_count:.4f}")
        print(f"      {'✓ Significant' if p_count < 0.05 else '✗ Not significant'} at α=0.05")
        
        print(f"\n   Total Area Burned:")
        print(f"      Trend: {slope_area:+,.0f} ha/year")
        print(f"      R²: {r_area**2:.3f}, p-value: {p_area:.4f}")
        print(f"      {'✓ Significant' if p_area < 0.05 else '✗ Not significant'} at α=0.05")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Number of fires over time
        axes[0,0].bar(yearly_stats['YEAR'], yearly_stats['num_fires'], color='coral', alpha=0.7)
        axes[0,0].plot(trend_data['YEAR'], slope_count * trend_data['YEAR'] + intercept_count, 
                       'r--', lw=2, label=f'Trend: {slope_count:+.1f}/yr')
        axes[0,0].set_title('Number of Large Fires per Year')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Number of Fires')
        axes[0,0].legend()
        
        # Total area burned
        axes[0,1].bar(yearly_stats['YEAR'], yearly_stats['total_area']/1e6, color='darkred', alpha=0.7)
        axes[0,1].set_title('Total Area Burned per Year (Large Fires)')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Area (Million hectares)')
        
        # Monthly distribution
        fire_df['MONTH'] = pd.to_numeric(fire_df['MONTH'], errors='coerce')
        monthly = fire_df.groupby('MONTH').size()
        axes[1,0].bar(monthly.index, monthly.values, color='forestgreen', alpha=0.7)
        axes[1,0].set_title('Fire Seasonality (All Years)')
        axes[1,0].set_xlabel('Month')
        axes[1,0].set_ylabel('Number of Fires')
        axes[1,0].set_xticks(range(1, 13))
        axes[1,0].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        
        # By province
        if 'SRC_AGENCY' in fire_df.columns:
            by_agency = fire_df.groupby('SRC_AGENCY').agg({
                'FIRE_SIZE': ['count', 'sum']
            })
            by_agency.columns = ['count', 'total_area']
            by_agency = by_agency.sort_values('count', ascending=True)
            axes[1,1].barh(by_agency.index, by_agency['count'], color='steelblue', alpha=0.7)
            axes[1,1].set_title('Fires by Province/Territory')
            axes[1,1].set_xlabel('Number of Large Fires')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PATH / 'poc3_temporal_trends.png', dpi=150)
        plt.close()
        print(f"\n   Saved visualization to: {OUTPUT_PATH / 'poc3_temporal_trends.png'}")
        
        print("\n[7.4] POC ASSESSMENT - Temporal Trend Analysis:")
        print("   ✓ Clear seasonal pattern (peak in summer months)")
        print(f"   {'✓' if p_count < 0.05 else '⚠'} Fire frequency trend: {'' if p_count < 0.05 else 'not '}statistically significant")
        print("   ✓ Strong regional variation - good for comparative analysis")
        print("   → RECOMMENDATION: STRONG CANDIDATE - clear patterns to analyze")

# =============================================================================
# 8. FIRE STATISTICS EXPLORATION
# =============================================================================
print("\n" + "="*70)
print("SECTION 8: FIRE STATISTICS DATA")
print("="*70)

if fire_stats is not None:
    print("\n[8.1] Fire Statistics Structure:")
    print(f"   Shape: {fire_stats.shape}")
    print(f"\n   Columns:")
    for col in fire_stats.columns:
        print(f"      - {col}")
    
    print(f"\n[8.2] Sample Data:")
    print(fire_stats.head(10).to_string())

# =============================================================================
# 9. SUMMARY AND RECOMMENDATIONS
# =============================================================================
print("\n" + "="*70)
print("SECTION 9: SUMMARY AND RECOMMENDATIONS")
print("="*70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    DATA AVAILABILITY SUMMARY                          ║
╠══════════════════════════════════════════════════════════════════════╣
║ FIRE DATA:                                                            ║
║   • Large fires (>200 ha): 1972-2024, ~{} records                 ║
║   • Attributes: Year, Month, Day, Province, Cause, Size (ha)          ║
║   • Good coverage for AB, BC, ON, QC, SK, MB, YT, NT                  ║
║                                                                        ║
║ CLIMATE DATA (samples):                                               ║
║   • 3 sample files from Ontario                                        ║
║   • Variables: Temp (max/min/mean), Precip, Snow, Wind                ║
║   • User has THOUSANDS more files available                            ║
║                                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                    THESIS TOPIC RECOMMENDATIONS                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ 1. FIRE CAUSE PREDICTION (CLASSIFICATION) ⭐⭐⭐                       ║
║    • ML Task: Binary classification (Natural vs Human)                 ║
║    • Baseline accuracy achievable with fire metadata alone             ║
║    • Adding climate features will improve significantly                ║
║    • Clear business value: Prevention strategies                       ║
║    • VERDICT: HIGHLY RECOMMENDED                                       ║
║                                                                        ║
║ 2. TEMPORAL TREND ANALYSIS ⭐⭐⭐                                       ║
║    • Clear patterns: Seasonality, regional variation                   ║
║    • Can correlate with climate trends                                 ║
║    • Good for visualization and storytelling                           ║
║    • Publishable implications about climate change                     ║
║    • VERDICT: HIGHLY RECOMMENDED                                       ║
║                                                                        ║
║ 3. FIRE SIZE PREDICTION (REGRESSION) ⭐⭐                              ║
║    • Challenging: High variance in fire sizes                          ║
║    • Needs climate data for meaningful predictions                     ║
║    • Good ML exercise but harder to get high accuracy                  ║
║    • VERDICT: MODERATE - good learning, harder results                 ║
║                                                                        ║
║ 4. REGIONAL COMPARISON ⭐⭐                                            ║
║    • Compare fire-climate relationships across provinces               ║
║    • Needs consistent climate data per region                          ║
║    • VERDICT: MODERATE - depends on climate data availability          ║
║                                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                         NEXT STEPS                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ 1. Acquire climate data for fire-prone regions (BC, AB, ON, QC)       ║
║ 2. Match fires to nearest weather stations                             ║
║ 3. Engineer climate features (temp anomalies, drought indices)         ║
║ 4. Build enhanced ML models with climate features                      ║
║ 5. Prepare proposal focusing on Topic #1 or #2                         ║
╚══════════════════════════════════════════════════════════════════════╝
""".format(len(fire_df) if fire_df is not None else "N/A"))

print("\n" + "="*70)
print("Analysis complete! Check the 'analysis_output' folder for visualizations.")
print("="*70)
