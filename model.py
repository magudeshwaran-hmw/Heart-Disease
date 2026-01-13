import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_model():
    print("🔄 Loading dataset...")
    try:
        df = pd.read_csv("heart.csv")
    except FileNotFoundError:
        print("❌ Error: heart.csv not found!")
        return

    # Check for missing values
    if df.isnull().sum().any():
        print("⚠️ Missing values found! Handling them...")
        df.fillna(df.mean(), inplace=True)

    # Features and Target
    X = df.drop(columns=['target'])
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dictionary to store models and their accuracies
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    print("\n🚀 Training Models...")
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"   🔹 {name}: {acc:.4f} accuracy")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name

    print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy:.4f} accuracy")

    # Save the best model and scaler
    with open("heart_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅ Model saved as 'heart_model.pkl'")
    print(f"✅ Scaler saved as 'scaler.pkl'")

if __name__ == "__main__":
    train_model()
