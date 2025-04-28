import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess the data
df = pd.read_csv('kidney_disease.csv')

# Replace empty strings and '?' with NaN
df = df.replace(['', '?'], pd.NA)

# Convert numerical columns to float
numerical_columns = [
    'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
    'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
]

categorical_columns = [
    'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

# Convert numerical columns
for col in numerical_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert categorical columns
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].fillna('unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Drop rows with NaN in numerical features
df = df.dropna(subset=numerical_columns)

# Convert target variable to binary
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

# Select all features
X = df[numerical_columns + categorical_columns]
y = df['classification']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the models
try:
    lg_clf = joblib.load('lg_clf.pkl')
    rf_clf = joblib.load('rf_clf.pkl')
    dt_clf = joblib.load('dt_clf.pkl')
    svm_clf = joblib.load('svm_clf.pkl')
    nb_clf = joblib.load('nb_clf.pkl')
    knn_clf = joblib.load('knn_clf.pkl')
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    print("Please make sure all model files exist in the current directory")
    raise

# Create a dictionary of models
models = {
    'Logistic Regression': lg_clf,
    'Random Forest': rf_clf,
    'Decision Tree': dt_clf,
    'SVM': svm_clf,
    'Naive Bayes': nb_clf,
    'KNN': knn_clf
}

# Create a dataframe to store the results
results = pd.DataFrame(columns=['Model', 'Accuracy'])

# Evaluate each model and store the results
for model_name, model in models.items():
    try:
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test_scaled)  # Use scaled test data
            accuracy = accuracy_score(y_test, y_pred)
            results = pd.concat([results, pd.DataFrame({'Model': [model_name], 'Accuracy': [accuracy]})], ignore_index=True)
        else:
            print(f"Model {model_name} is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")

# Print the results
print("\nModel Performance Results:")
print(results)

# Select the best model
if not results.empty:
    best_model = results.loc[results['Accuracy'].idxmax()]['Model']
    print("\nThe best model is:", best_model)
else:
    print("\nNo models were successfully evaluated.") 