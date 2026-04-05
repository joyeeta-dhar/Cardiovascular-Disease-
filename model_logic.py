import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

class CardioModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            max_depth=9, 
            max_features='sqrt', 
            min_samples_leaf=4, 
            min_samples_split=4, 
            n_estimators=190,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def preprocess_data(self, df):
        """Applies feature engineering as per the notebook logic."""
        df = df.copy()
        
        # BMI Calculation
        if 'bmi' not in df.columns:
            df['bmi'] = df['weight'] / (df['height'] / 100)**2
        
        # Age in years (Matching notebook .astype(int) logic)
        if 'age_years' not in df.columns:
            df['age_years'] = (df['age'] / 365.25).astype(int)

        # Blood Pressure Category Mapping (Notebook Logic)
        def get_bp_category(row):
            hi, lo = row['ap_hi'], row['ap_lo']
            if hi < 120 and lo < 80: return 0  # Normal
            if 120 <= hi < 130 and lo < 80: return 1  # Elevated
            if 130 <= hi < 140 or 80 <= lo < 90: return 2  # Stage 1
            return 3  # Stage 2

        # We always recalculate this to ensure numeric types (0-3) for our logic
        df['bp_category'] = df.apply(get_bp_category, axis=1)

        # Pulse Pressure
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

        # Hypertension Score
        df['hyper_score'] = df['bmi'] * df['cholesterol'] / df['gluc']

        # Obesity
        df['is_obese'] = (df['bmi'] > 30).astype(int)

        # Age Groups
        df['age_group'] = pd.cut(
            df['age_years'], 
            bins=[20, 30, 40, 50, 60, 80], 
            labels=[1, 2, 3, 4, 5]
        ).astype(float).fillna(-1).astype(int)

        return df

    def train(self, df):
        """Trains the model using SMOTE and Random Forest."""
        df = self.preprocess_data(df)
        
        features = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active',
            'age_years', 'bmi', 'bp_category'
        ]
        
        X = df[features]
        y = df['cardio']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # SMOTE for balance
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        # Scale
        X_train_res = self.scaler.fit_transform(X_train_res)
        
        # Fit
        self.model.fit(X_train_res, y_train_res)
        self.is_trained = True
        
        return self.model

    def predict(self, input_data):
        """Predicts risk for a single patient."""
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        # input_data should be a dict or single row df
        df_input = pd.DataFrame([input_data])
        df_processed = self.preprocess_data(df_input)
        
        features = [
            'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active',
            'age_years', 'bmi', 'bp_category'
        ]
        
        # Use RAW features (no scaling) to match notebook's Random Forest training
        X_input = df_processed[features]
        prob = self.model.predict_proba(X_input)[0][1]
        pred = self.model.predict(X_input)[0]
        
        return pred, prob

    def save_model(self, path="model.joblib"):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)

    def load_model(self, path="model.joblib"):
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True
            return True
        return False
