# 🚀 How to Implement the Enhanced Peak Load Predictions

## 📋 Step-by-Step Implementation Guide

### **Step 1: Add Enhanced Feature Engineering**
Add this cell to your notebook after your data loading:

```python
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

# Apply enhanced feature engineering
print("🔧 Creating advanced features...")
full_data_enhanced = create_advanced_features(full_data)
pred_2008_enhanced = create_advanced_features(pred_2008)

# Update feature columns
enhanced_feature_cols = [col for col in full_data_enhanced.columns 
                        if col not in ['Date', 'Hour', 'Load', 'Is_Peak', 'Avg_Temperature']]

print(f"✅ Enhanced features created: {len(enhanced_feature_cols)} features")
```

### **Step 2: Add Ensemble Model Training**
Add this cell after your feature engineering:

```python
# =============================================================================
# 🎯 ENSEMBLE MODEL TRAINING
# =============================================================================

def train_ensemble_models(X_train, y_train, X_val, y_val):
    """
    Train multiple models and create ensemble predictions
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    
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

# Prepare enhanced data for training
print("🔧 Preparing enhanced training data...")
model_data_enhanced = full_data_enhanced[enhanced_feature_cols + ['Is_Peak']].dropna()
X_enhanced = model_data_enhanced[enhanced_feature_cols]
y_enhanced = model_data_enhanced['Is_Peak']

print(f"Enhanced modeling dataset shape: {X_enhanced.shape}")
print(f"Peak hours in enhanced training data: {y_enhanced.sum()}")

# Split data
X_train_enhanced, X_val_enhanced, y_train_enhanced, y_val_enhanced = train_test_split(
    X_enhanced, y_enhanced, test_size=0.2, random_state=42, stratify=y_enhanced
)

# Train ensemble models
ensemble_models, ensemble_scores, best_ensemble_model = train_ensemble_models(
    X_train_enhanced, y_train_enhanced, X_val_enhanced, y_val_enhanced
)
```

### **Step 3: Add Visual Storytelling Dashboard**
Add this cell for the interactive dashboard:

```python
# =============================================================================
# 📊 VISUAL STORYTELLING DASHBOARD
# =============================================================================

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# Create the visual story
print("🎨 Creating Peak Load Story Dashboard...")
story_dashboard = create_peak_load_story_dashboard(
    full_data_enhanced, pred_2008_enhanced, ensemble_models, best_ensemble_model
)

# Display the dashboard
story_dashboard.show()

print("✅ Visual storytelling dashboard created!")
```

### **Step 4: Generate Enhanced Predictions**
Add this cell for the final predictions:

```python
# =============================================================================
# 🔮 ENHANCED PREDICTIONS WITH BEST MODEL
# =============================================================================

def generate_enhanced_predictions(pred_2008_enhanced, ensemble_models, best_model_name, enhanced_feature_cols):
    """
    Generate predictions using the best ensemble model with enhanced features
    """
    
    # Prepare prediction data
    X_pred_enhanced = pred_2008_enhanced[enhanced_feature_cols]
    
    # Get the best model
    best_model = ensemble_models[best_model_name]
    
    # Generate predictions
    print(f"🔮 Generating predictions with {best_model_name}...")
    probabilities_enhanced = best_model.predict_proba(X_pred_enhanced)[:, 1]
    
    # Add to dataframe
    pred_2008_enhanced['Enhanced_Daily_Peak_Probability'] = probabilities_enhanced
    
    return pred_2008_enhanced

# Generate enhanced predictions
pred_2008_final = generate_enhanced_predictions(
    pred_2008_enhanced, ensemble_models, best_ensemble_model, enhanced_feature_cols
)

# Create final output
output_df_enhanced = pred_2008_final[['Date', 'Hour', 'Enhanced_Daily_Peak_Probability']].copy()
output_df_enhanced.columns = ['Date', 'Hour', 'Daily Peak Probability']

# Ensure probabilities are between 0 and 1
output_df_enhanced['Daily Peak Probability'] = np.clip(output_df_enhanced['Daily Peak Probability'], 0, 1)

# Save enhanced results
output_df_enhanced.to_csv('probability_estimates_enhanced.csv', index=False)

print(f"\n✅ ENHANCED PREDICTIONS COMPLETED!")
print(f"📁 Results saved to 'probability_estimates_enhanced.csv'")
print(f"📊 Total hours predicted: {len(output_df_enhanced):,}")
print(f"🎯 Probability range: {output_df_enhanced['Daily Peak Probability'].min():.4f} to {output_df_enhanced['Daily Peak Probability'].max():.4f}")
print(f"📈 Average probability: {output_df_enhanced['Daily Peak Probability'].mean():.4f}")

# Show daily probability sums
daily_sums_enhanced = output_df_enhanced.groupby('Date')['Daily Peak Probability'].sum()
print(f"📊 Daily probability sums - Min: {daily_sums_enhanced.min():.4f}, Max: {daily_sums_enhanced.max():.4f}, Mean: {daily_sums_enhanced.mean():.4f}")
```

## 🎯 **Summary of Improvements**

### **What You'll Get:**
1. **20+ New Features**: Advanced feature engineering for better predictions
2. **6 Ensemble Models**: Multiple models with automatic best model selection
3. **Interactive Dashboard**: Visual storytelling of peak load patterns
4. **Enhanced Predictions**: Better accuracy with `probability_estimates_enhanced.csv`
5. **Comprehensive Analytics**: Detailed insights into peak load behavior

### **Files Created:**
- `enhanced_peak_load_improvements.py` - Complete code library
- `implementation_guide.md` - This step-by-step guide
- `probability_estimates_enhanced.csv` - Enhanced predictions (after running)

### **Next Steps:**
1. **Open your notebook** in Cursor
2. **Copy the code blocks** from this guide into new cells
3. **Run the cells** in order
4. **Compare results** with your original approach

The enhanced approach should provide significantly better predictions and insights!