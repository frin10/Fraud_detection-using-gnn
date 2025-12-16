import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Load your dataset
# df = pd.read_csv('your_fraud_data.csv')
# For demo, I'll show you the complete process

def balance_fraud_dataset(df, target_size=10000, fraud_ratio=0.3):
    """
    Balance fraud dataset to specified size with desired fraud ratio
    
    Parameters:
    - df: Your original dataframe
    - target_size: Total number of records you want (default: 10000)
    - fraud_ratio: Proportion of fraud cases (0.3 = 30% fraud)
    """
    
    print("="*60)
    print("FRAUD DATASET BALANCING")
    print("="*60)
    
    # Check original distribution
    fraud_col = 'isFraud'  # Based on your screenshot
    original_fraud_count = df[fraud_col].sum()
    original_total = len(df)
    
    print(f"\n📊 Original Dataset:")
    print(f"   Total records: {original_total}")
    print(f"   Fraud cases: {original_fraud_count} ({original_fraud_count/original_total*100:.2f}%)")
    print(f"   Non-fraud cases: {original_total - original_fraud_count} ({(original_total-original_fraud_count)/original_total*100:.2f}%)")
    
    # Separate features and target
    X = df.drop([fraud_col, 'isFlaggedFraud'], axis=1, errors='ignore')
    y = df[fraud_col]
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Calculate target counts
    target_fraud = int(target_size * fraud_ratio)
    target_non_fraud = target_size - target_fraud
    
    print(f"\n🎯 Target Dataset:")
    print(f"   Total records: {target_size}")
    print(f"   Target fraud cases: {target_fraud} ({fraud_ratio*100:.0f}%)")
    print(f"   Target non-fraud cases: {target_non_fraud} ({(1-fraud_ratio)*100:.0f}%)")
    
    # Strategy 1: Using SMOTE
    print(f"\n🔄 Applying SMOTE augmentation...")
    
    # Define sampling strategy
    sampling_strategy = {1: target_fraud, 0: target_non_fraud}
    
    try:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X_encoded, y)
        
        print(f"✅ SMOTE completed successfully!")
        print(f"   Balanced dataset size: {len(X_balanced)}")
        print(f"   Fraud cases: {y_balanced.sum()}")
        print(f"   Non-fraud cases: {len(y_balanced) - y_balanced.sum()}")
        
        # Create balanced dataframe
        df_balanced = pd.DataFrame(X_balanced, columns=X_encoded.columns)
        df_balanced[fraud_col] = y_balanced
        
        # Decode categorical columns back
        for col in categorical_cols:
            if col in df_balanced.columns:
                # Handle out-of-range values from synthetic data
                df_balanced[col] = df_balanced[col].round().astype(int)
                max_val = len(label_encoders[col].classes_) - 1
                df_balanced[col] = df_balanced[col].clip(0, max_val)
                df_balanced[col] = label_encoders[col].inverse_transform(df_balanced[col])
        
        return df_balanced
        
    except Exception as e:
        print(f"❌ Error with SMOTE: {str(e)}")
        print("   Trying alternative approach...")
        return balance_with_oversampling(df, target_size, fraud_ratio)


def balance_with_oversampling(df, target_size=10000, fraud_ratio=0.3):
    """
    Alternative method: Manual oversampling with noise
    """
    fraud_col = 'isFraud'
    
    # Separate fraud and non-fraud
    fraud_data = df[df[fraud_col] == 1].copy()
    non_fraud_data = df[df[fraud_col] == 0].copy()
    
    target_fraud = int(target_size * fraud_ratio)
    target_non_fraud = target_size - target_fraud
    
    # Oversample fraud cases
    if len(fraud_data) < target_fraud:
        # Calculate how many times to repeat and add noise
        times = target_fraud // len(fraud_data)
        remainder = target_fraud % len(fraud_data)
        
        augmented_fraud = [fraud_data]
        
        for i in range(times - 1):
            noisy_fraud = fraud_data.copy()
            # Add small noise to numerical columns
            numerical_cols = noisy_fraud.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col != fraud_col and col != 'isFlaggedFraud':
                    noise = np.random.normal(0, 0.02 * noisy_fraud[col].std(), size=len(noisy_fraud))
                    noisy_fraud[col] = noisy_fraud[col] + noise
            augmented_fraud.append(noisy_fraud)
        
        # Add remainder
        if remainder > 0:
            augmented_fraud.append(fraud_data.sample(n=remainder, replace=True, random_state=42))
        
        fraud_balanced = pd.concat(augmented_fraud, ignore_index=True)
    else:
        fraud_balanced = fraud_data.sample(n=target_fraud, random_state=42)
    
    # Sample non-fraud cases
    if len(non_fraud_data) >= target_non_fraud:
        non_fraud_balanced = non_fraud_data.sample(n=target_non_fraud, random_state=42)
    else:
        non_fraud_balanced = non_fraud_data.sample(n=target_non_fraud, replace=True, random_state=42)
    
    # Combine
    df_balanced = pd.concat([fraud_balanced, non_fraud_balanced], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Manual oversampling completed!")
    print(f"   Balanced dataset size: {len(df_balanced)}")
    print(f"   Fraud cases: {df_balanced[fraud_col].sum()}")
    
    return df_balanced


# USAGE EXAMPLE:
# ===============
# 1. Load your data
df = pd.read_csv('AIML Dataset.csv')

# 2. Balance the dataset
df_balanced = balance_fraud_dataset(df, target_size=10000, fraud_ratio=0.3)

# 3. Save the balanced dataset
df_balanced.to_csv('fraud_data_balanced_10k.csv', index=False)

# 4. Verify the results
print("\n" + "="*60)
print("FINAL BALANCED DATASET")
print("="*60)
print(df_balanced['isFraud'].value_counts())
print(f"\nFraud percentage: {df_balanced['isFraud'].mean()*100:.2f}%")


# RECOMMENDED FRAUD RATIOS:
# =========================
# - 20-30% fraud: Good for most models, maintains some class imbalance
# - 30-40% fraud: More aggressive balancing
# - 50% fraud: Perfect balance (use with caution - may not reflect reality)
