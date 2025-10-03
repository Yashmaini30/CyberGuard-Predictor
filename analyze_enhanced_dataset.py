"""
Enhanced Cybercrime Dataset Analysis
Analyze the structure and features for ML model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

def analyze_enhanced_dataset():
    """Analyze the enhanced cybercrime dataset for ML feature engineering"""
    
    print("🔍 Analyzing Enhanced Cybercrime Dataset")
    print("=" * 60)
    
    # Load the enhanced dataset
    try:
        df = pd.read_csv('enhanced_indian_cybercrime_data.csv')
        print(f"✅ Dataset loaded successfully: {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Basic dataset info
    print(f"\n📊 Dataset Overview:")
    print(f"   Records: {len(df):,}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display columns and types
    print(f"\n📋 Column Information:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        print(f"   {col:25} | {str(dtype):12} | Nulls: {null_count:5} | Unique: {unique_count:6}")
    
    # Target variables analysis
    print(f"\n🎯 Target Variables Analysis:")
    
    # Risk Score distribution
    print(f"\n   Risk Score:")
    print(f"     Range: {df['risk_score'].min():.1f} - {df['risk_score'].max():.1f}")
    print(f"     Mean: {df['risk_score'].mean():.2f}")
    print(f"     Std: {df['risk_score'].std():.2f}")
    
    # Withdrawal Probability distribution
    print(f"\n   Withdrawal Probability:")
    print(f"     Range: {df['withdrawal_probability'].min():.3f} - {df['withdrawal_probability'].max():.3f}")
    print(f"     Mean: {df['withdrawal_probability'].mean():.3f}")
    print(f"     Std: {df['withdrawal_probability'].std():.3f}")
    
    # Categorical features analysis
    print(f"\n📊 Categorical Features:")
    
    categorical_features = ['crime_type', 'urgency_level', 'complaint_state', 'status', 'reported_by', 'bank_involved']
    
    for feature in categorical_features:
        if feature in df.columns:
            print(f"\n   {feature}:")
            value_counts = df[feature].value_counts()
            for i, (value, count) in enumerate(value_counts.head(5).items()):
                percentage = (count / len(df)) * 100
                print(f"     {value:30} | {count:6} ({percentage:5.1f}%)")
            if len(value_counts) > 5:
                print(f"     ... and {len(value_counts) - 5} more categories")
    
    # Numerical features analysis
    print(f"\n💰 Numerical Features:")
    
    numerical_features = ['amount_lost', 'complaint_lat', 'complaint_lng', 'predicted_withdrawal_lat', 'predicted_withdrawal_lng']
    
    for feature in numerical_features:
        if feature in df.columns:
            print(f"\n   {feature}:")
            print(f"     Range: {df[feature].min():.2f} - {df[feature].max():.2f}")
            print(f"     Mean: {df[feature].mean():.2f}")
            print(f"     Std: {df[feature].std():.2f}")
            print(f"     Nulls: {df[feature].isnull().sum()}")
    
    # Temporal analysis
    print(f"\n⏰ Temporal Analysis:")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"     Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"     Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        print(f"\n     Peak hours: {df['hour'].value_counts().head(3).index.tolist()}")
        print(f"     Peak days: {df['day_of_week'].value_counts().head(3).index.tolist()}")
        print(f"     Peak months: {df['month'].value_counts().head(3).index.tolist()}")
    
    # Geographical distribution
    print(f"\n🗺️ Geographical Distribution:")
    
    state_stats = df.groupby('complaint_state').agg({
        'complaint_id': 'count',
        'amount_lost': ['sum', 'mean'],
        'risk_score': 'mean',
        'withdrawal_probability': 'mean'
    }).round(2)
    
    state_stats.columns = ['Cases', 'Total_Loss', 'Avg_Loss', 'Avg_Risk', 'Avg_Withdrawal_Prob']
    state_stats = state_stats.sort_values('Cases', ascending=False)
    
    print(f"\n   Top 10 States by Cases:")
    for state, row in state_stats.head(10).iterrows():
        print(f"     {state:20} | {row['Cases']:5} cases | ₹{row['Total_Loss']:12,.0f} total | Risk: {row['Avg_Risk']:4.1f}")
    
    # Feature correlation analysis
    print(f"\n🔗 Feature Correlations with Target Variables:")
    
    # Prepare numerical features for correlation
    numerical_df = df.select_dtypes(include=[np.number]).copy()
    
    if len(numerical_df.columns) > 2:
        # Correlation with risk_score
        risk_corr = numerical_df.corr()['risk_score'].sort_values(ascending=False)
        print(f"\n   Correlation with Risk Score:")
        for feature, corr in risk_corr.items():
            if feature != 'risk_score' and abs(corr) > 0.1:
                print(f"     {feature:25} | {corr:6.3f}")
        
        # Correlation with withdrawal_probability
        withdrawal_corr = numerical_df.corr()['withdrawal_probability'].sort_values(ascending=False)
        print(f"\n   Correlation with Withdrawal Probability:")
        for feature, corr in withdrawal_corr.items():
            if feature != 'withdrawal_probability' and abs(corr) > 0.1:
                print(f"     {feature:25} | {corr:6.3f}")
    
    # Feature engineering recommendations
    print(f"\n💡 Feature Engineering Recommendations:")
    print(f"   1. Encode categorical variables: {categorical_features}")
    print(f"   2. Create temporal features: hour, day_of_week, month, is_weekend")
    print(f"   3. Geographic features: distance_to_major_city, state_risk_factor")
    print(f"   4. Amount features: log_amount, amount_category (small/medium/large)")
    print(f"   5. Composite features: crime_severity_score, location_risk_index")
    
    # Model training recommendations
    print(f"\n🤖 Model Training Recommendations:")
    print(f"   Target Variables:")
    print(f"     - Primary: risk_score (regression)")
    print(f"     - Secondary: withdrawal_probability (regression)")
    print(f"     - Tertiary: urgency_level (classification)")
    print(f"   ")
    print(f"   Model Types:")
    print(f"     - Random Forest (good for mixed data types)")
    print(f"     - Gradient Boosting (XGBoost/LightGBM)")
    print(f"     - Neural Networks (for complex patterns)")
    print(f"   ")
    print(f"   Validation Strategy:")
    print(f"     - Time-based split (older data for training)")
    print(f"     - Cross-validation with temporal ordering")
    print(f"     - State-wise stratification")
    
    return df

if __name__ == "__main__":
    df = analyze_enhanced_dataset()
    print(f"\n✅ Analysis complete! Dataset ready for ML model training.")