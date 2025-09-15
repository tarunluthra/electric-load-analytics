# =============================================================================
# 🚀 ENHANCED PEAK LOAD PREDICTION IMPROVEMENTS
# =============================================================================
# This file contains all the improvements I suggested for your notebook
# You can copy these into new cells in your notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("🎯 ENHANCED PEAK LOAD PREDICTION FRAMEWORK")
print("=" * 60)

# =============================================================================
# 📊 ENHANCED FEATURE ENGINEERING
# =============================================================================

def create_advanced_features(df):
    """
    Create advanced features for better peak load prediction
    """
    df = df.copy()
    
    # 1. TEMPORAL FEATURES
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    # 2. WEATHER INTERACTION FEATURES
    df['Temp_Hour_Interaction'] = df['Avg_Temperature'] * df['Hour']
    df['Temp_Weekend_Interaction'] = df['Avg_Temperature'] * df['IsWeekend']
    df['Temp_Holiday_Interaction'] = df['Avg_Temperature'] * df['IsHoliday']
    
    # 3. LOAD PATTERN FEATURES
    df['Is_Peak_Hour_Range'] = df['Hour'].isin([17, 18, 19, 20]).astype(int)
    df['Is_Morning_Rush'] = df['Hour'].isin([7, 8, 9]).astype(int)
    df['Is_Evening_Rush'] = df['Hour'].isin([17, 18, 19]).astype(int)
    
    # 4. TEMPERATURE EXTREMES
    df['Is_Hot_Day'] = (df['Avg_Temperature'] > df['Avg_Temperature'].quantile(0.8)).astype(int)
    df['Is_Cold_Day'] = (df['Avg_Temperature'] < df['Avg_Temperature'].quantile(0.2)).astype(int)
    
    # 5. SEASONAL FEATURES
    df['Is_Summer'] = df['Month'].isin([6, 7, 8]).astype(int)
    df['Is_Winter'] = df['Month'].isin([12, 1, 2]).astype(int)
    df['Is_Spring'] = df['Month'].isin([3, 4, 5]).astype(int)
    df['Is_Fall'] = df['Month'].isin([9, 10, 11]).astype(int)
    
    # 6. COOLING/HEATING DEGREE FEATURES
    df['Cooling_Degree_Hours'] = np.maximum(0, df['Avg_Temperature'] - 75)
    df['Heating_Degree_Hours'] = np.maximum(0, 65 - df['Avg_Temperature'])
    
    return df

# =============================================================================
# 🎯 ENSEMBLE MODEL TRAINING
# =============================================================================

def train_ensemble_models(X_train, y_train, X_val, y_val):
    """
    Train multiple models and create ensemble predictions
    """
    models = {}
    
    # 1. RANDOM FOREST
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    
    # 2. XGBOOST
    print("🚀 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # 3. GRADIENT BOOSTING
    print("📈 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['GradientBoosting'] = gb
    
    # 4. LOGISTIC REGRESSION
    print("📊 Training Logistic Regression...")
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)
    models['LogisticRegression'] = lr
    
    # 5. ENSEMBLE - VOTING CLASSIFIER
    print("🗳️ Creating Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb_model),
            ('gb', gb),
            ('lr', lr)
        ],
        voting='soft'  # Use probabilities
    )
    voting_clf.fit(X_train, y_train)
    models['VotingEnsemble'] = voting_clf
    
    # 6. ENSEMBLE - STACKING CLASSIFIER
    print("🏗️ Creating Stacking Ensemble...")
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb_model),
            ('gb', gb)
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )
    stacking_clf.fit(X_train, y_train)
    models['StackingEnsemble'] = stacking_clf
    
    # Evaluate all models
    print("\n📊 MODEL EVALUATION:")
    print("-" * 40)
    
    model_scores = {}
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        brier_score = brier_score_loss(y_val, y_pred_proba)
        model_scores[name] = brier_score
        print(f"{name:20s}: Brier Score = {brier_score:.4f}")
    
    best_model_name = min(model_scores, key=model_scores.get)
    print(f"\n🏆 Best Model: {best_model_name} (Brier Score: {model_scores[best_model_name]:.4f})")
    
    return models, model_scores, best_model_name

# =============================================================================
# 📊 VISUAL STORYTELLING DASHBOARD
# =============================================================================

def create_peak_load_story_dashboard(full_data, pred_2008, ensemble_models, best_model_name):
    """
    Create a comprehensive visual story of peak load patterns and predictions
    """
    
    # 1. THE PEAK LOAD STORY: WHEN DO PEAKS HAPPEN?
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            '🌅 The Daily Peak Load Journey',
            '🌡️ Temperature vs Peak Load Relationship', 
            '📅 Seasonal Peak Patterns',
            '🏠 Weekend vs Weekday Peak Behavior',
            '🎯 Model Performance Comparison',
            '🔮 2008 Peak Predictions Heatmap'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Story 1: Daily Peak Journey
    hourly_peaks = full_data.groupby('Hour')['Is_Peak'].mean()
    fig.add_trace(
        go.Scatter(
            x=hourly_peaks.index,
            y=hourly_peaks.values,
            mode='lines+markers',
            name='Peak Probability',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8, color='#FF6B6B')
        ),
        row=1, col=1
    )
    
    # Story 2: Temperature Impact
    temp_bins = pd.cut(full_data['Avg_Temperature'], bins=10)
    temp_peak_rates = full_data.groupby(temp_bins)['Is_Peak'].mean()
    temp_centers = [interval.mid for interval in temp_peak_rates.index]
    
    fig.add_trace(
        go.Scatter(
            x=temp_centers,
            y=temp_peak_rates.values,
            mode='lines+markers',
            name='Peak Rate by Temperature',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8, color='#4ECDC4')
        ),
        row=1, col=2
    )
    
    # Story 3: Seasonal Patterns
    monthly_peaks = full_data.groupby('Month')['Is_Peak'].mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.add_trace(
        go.Bar(
            x=month_names,
            y=monthly_peaks.values,
            name='Monthly Peak Rate',
            marker_color='#45B7D1'
        ),
        row=2, col=1
    )
    
    # Story 4: Weekend vs Weekday
    weekend_peaks = full_data.groupby('IsWeekend')['Is_Peak'].mean()
    fig.add_trace(
        go.Bar(
            x=['Weekday', 'Weekend'],
            y=weekend_peaks.values,
            name='Peak Rate by Day Type',
            marker_color=['#96CEB4', '#FFEAA7']
        ),
        row=2, col=2
    )
    
    # Story 5: Model Performance
    model_names = list(ensemble_models.keys())
    model_scores = [ensemble_scores[name] for name in model_names]
    
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=model_scores,
            name='Brier Score (Lower is Better)',
            marker_color=['#FF6B6B' if name == best_model_name else '#D3D3D3' for name in model_names]
        ),
        row=3, col=1
    )
    
    # Story 6: 2008 Predictions Heatmap
    if 'pred_2008' in locals():
        # Create heatmap data
        heatmap_data = pred_2008.pivot_table(
            values='Daily Peak Probability', 
            index='Month', 
            columns='Hour', 
            aggfunc='mean'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Reds',
                name='Peak Probability Heatmap'
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': '🏭 The Peak Load Prediction Story: Understanding Energy Demand Patterns',
            'x': 0.5,
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        height=1200,
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='white'
    )
    
    return fig

# =============================================================================
# 📈 ADVANCED ANALYTICS & INSIGHTS
# =============================================================================

def generate_peak_load_insights(full_data_enhanced, output_df_enhanced, ensemble_models, best_model_name):
    """
    Generate comprehensive insights and analytics about peak load patterns
    """
    
    print("🔍 GENERATING PEAK LOAD INSIGHTS")
    print("=" * 50)
    
    # 1. PEAK HOUR ANALYSIS
    print("\n🌅 PEAK HOUR ANALYSIS:")
    print("-" * 25)
    
    hourly_peaks = full_data_enhanced.groupby('Hour')['Is_Peak'].mean().sort_values(ascending=False)
    top_peak_hours = hourly_peaks.head(5)
    
    print("Top 5 Peak Hours (Historical):")
    for hour, prob in top_peak_hours.items():
        print(f"  Hour {hour:2d}: {prob:.3f} ({prob*100:.1f}%)")
    
    # 2. TEMPERATURE IMPACT ANALYSIS
    print("\n🌡️ TEMPERATURE IMPACT ANALYSIS:")
    print("-" * 35)
    
    # Temperature bins
    temp_bins = pd.cut(full_data_enhanced['Avg_Temperature'], bins=5)
    temp_analysis = full_data_enhanced.groupby(temp_bins).agg({
        'Is_Peak': ['mean', 'count'],
        'Avg_Temperature': 'mean'
    }).round(3)
    
    print("Temperature vs Peak Rate:")
    for i, (temp_range, row) in enumerate(temp_analysis.iterrows()):
        temp_avg = row[('Avg_Temperature', 'mean')]
        peak_rate = row[('Is_Peak', 'mean')]
        count = row[('Is_Peak', 'count')]
        print(f"  {temp_range}: {peak_rate:.3f} ({peak_rate*100:.1f}%) - {count} observations")
    
    # 3. SEASONAL PATTERNS
    print("\n📅 SEASONAL PEAK PATTERNS:")
    print("-" * 30)
    
    monthly_peaks = full_data_enhanced.groupby('Month')['Is_Peak'].mean().sort_values(ascending=False)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print("Peak Rates by Month:")
    for month, prob in monthly_peaks.items():
        print(f"  {month_names[month-1]:3s}: {prob:.3f} ({prob*100:.1f}%)")
    
    # 4. WEEKEND VS WEEKDAY
    print("\n🏠 WEEKEND VS WEEKDAY ANALYSIS:")
    print("-" * 35)
    
    weekend_analysis = full_data_enhanced.groupby('IsWeekend')['Is_Peak'].mean()
    print(f"Weekday Peak Rate: {weekend_analysis[0]:.3f} ({weekend_analysis[0]*100:.1f}%)")
    print(f"Weekend Peak Rate: {weekend_analysis[1]:.3f} ({weekend_analysis[1]*100:.1f}%)")
    print(f"Difference: {abs(weekend_analysis[1] - weekend_analysis[0]):.3f}")
    
    return {
        'hourly_peaks': hourly_peaks,
        'temperature_analysis': temp_analysis,
        'monthly_peaks': monthly_peaks,
        'weekend_analysis': weekend_analysis
    }

# =============================================================================
# 🧪 ROBUST VALIDATION & EVALUATION FRAMEWORK
# =============================================================================

def create_validation_framework(X, y, models, feature_cols):
    """
    Create a comprehensive validation framework for model evaluation
    """
    
    print("🧪 CREATING VALIDATION FRAMEWORK")
    print("=" * 40)
    
    # 1. TIME SERIES CROSS-VALIDATION
    print("\n📅 TIME SERIES CROSS-VALIDATION:")
    print("-" * 35)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    validation_results = {}
    
    for name, model in models.items():
        print(f"\n🔍 Validating {name}...")
        
        # Time series CV scores
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_brier_score')
        cv_scores = -cv_scores  # Convert back to positive Brier scores
        
        validation_results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"  CV Brier Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  CV Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    
    return validation_results

print("✅ ENHANCED PEAK LOAD PREDICTION FRAMEWORK LOADED!")
print("📋 Copy the functions above into your notebook cells to implement the improvements")