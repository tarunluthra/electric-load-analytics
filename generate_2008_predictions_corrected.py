# =============================================================================
# 6. GENERATE 2008 PREDICTIONS (CORRECTED VERSION)
# =============================================================================

def generate_2008_predictions(weather_df, target_df, best_model, trained_models, scalers, feature_columns):
    """Generate 2008 predictions with proper calibration"""
    print("🔮 Generating 2008 predictions...")
    
    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(weather_df['Date']):
        weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    
    # Load 2008 weather data - FIXED: Use .dt.year instead of .str.contains
    weather_2008 = weather_df[weather_df['Date'].dt.year == 2008]
    weather_2008_agg = weather_2008.groupby(['Date', 'Hour'])['Temperature'].mean().reset_index()
    weather_2008_agg.columns = ['Date', 'Hour', 'Avg_Temperature']
    
    # Create 2008 prediction dataframe
    pred_2008 = target_df.copy()
    pred_2008['Date'] = pd.to_datetime(pred_2008['Date'])
    
    # Merge with weather data
    pred_2008 = pred_2008.merge(weather_2008_agg, on=['Date', 'Hour'], how='left')
    
    # Fill missing temperature data
    hourly_temp_avg = weather_df.groupby('Hour')['Temperature'].mean()
    pred_2008['Avg_Temperature'] = pred_2008['Avg_Temperature'].fillna(
        pred_2008['Hour'].map(hourly_temp_avg)
    )
    
    # Add features (simplified version)
    pred_2008['Month'] = pred_2008['Date'].dt.month
    pred_2008['DayOfWeek'] = pred_2008['Date'].dt.dayofweek
    pred_2008['DayOfYear'] = pred_2008['Date'].dt.dayofyear
    pred_2008['IsWeekend'] = (pred_2008['DayOfWeek'] >= 5).astype(int)
    pred_2008['Hour_Sin'] = np.sin(2 * np.pi * pred_2008['Hour'] / 24)
    pred_2008['Hour_Cos'] = np.cos(2 * np.pi * pred_2008['Hour'] / 24)
    pred_2008['Temp_Squared'] = pred_2008['Avg_Temperature'] ** 2
    pred_2008['Temp_Cubed'] = pred_2008['Avg_Temperature'] ** 3
    
    # Holiday features
    years = pred_2008['Date'].dt.year.unique()
    all_holidays = []
    for year in years:
        holidays = [f"{year}-01-01", f"{year}-07-04", f"{year}-12-25"]
        all_holidays.extend(holidays)
    pred_2008['IsHoliday'] = pred_2008['Date'].astype(str).isin(all_holidays).astype(int)
    
    # Add default values for missing features
    pred_2008['Temp_Std'] = 0
    pred_2008['Temp_Min'] = pred_2008['Avg_Temperature']
    pred_2008['Temp_Max'] = pred_2008['Avg_Temperature']
    pred_2008['Prev_Day_Temp'] = pred_2008['Avg_Temperature']
    pred_2008['Prev_Day_Load'] = 0
    pred_2008['Temp_Diff'] = 0
    
    # Generate predictions
    X_pred = pred_2008[feature_columns].fillna(0)
    
    if best_model == 'Logistic Regression':
        X_pred_scaled = scalers[best_model].transform(X_pred)
        probabilities = trained_models[best_model].predict_proba(X_pred_scaled)[:, 1]
    else:
        probabilities = trained_models[best_model].predict_proba(X_pred)[:, 1]
    
    # Apply calibration to fix daily sums
    pred_2008['Daily_Peak_Probability'] = probabilities * 0.8  # Reduce overconfidence
    
    # Normalize daily sums to 1.0
    daily_sums = pred_2008.groupby('Date')['Daily_Peak_Probability'].sum()
    pred_2008['Daily_Peak_Probability'] = pred_2008.apply(
        lambda row: row['Daily_Peak_Probability'] / daily_sums[row['Date']], axis=1
    )
    
    # Validate results
    daily_sums_final = pred_2008.groupby('Date')['Daily_Peak_Probability'].sum()
    print(f"Daily probability sums - Min: {daily_sums_final.min():.4f}, Max: {daily_sums_final.max():.4f}, Mean: {daily_sums_final.mean():.4f}")
    
    print("✅ 2008 predictions generated with proper calibration")
    return pred_2008

# Generate predictions
# pred_2008 = generate_2008_predictions(weather_df, target_df, best_model, trained_models, scalers, feature_columns)