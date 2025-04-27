import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv('kidney_disease.csv')
    
    # Replace empty strings and '?' with NaN
    df = df.replace(['', '?'], pd.NA)
    
    # Convert numerical columns to float
    numerical_columns = [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
        'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
    ]
    
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in numerical features
    df = df.dropna(subset=numerical_columns)
    
    # Convert target variable to binary
    df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
    
    # Select features and target
    X = df[numerical_columns]
    y = df['classification']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Train models and get predictions
def train_and_predict(X_train, X_test, y_train):
    # Initialize models
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42)
    lr = LogisticRegression(random_state=42)
    svm = SVC(random_state=42)
    nb = GaussianNB()
    knn = KNeighborsClassifier()
    
    # Train models
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    
    # Get predictions
    predDT = dt.predict(X_test)
    predRFC = rf.predict(X_test)
    predlog = lr.predict(X_test)
    predSVC = svm.predict(X_test)
    predgnb = nb.predict(X_test)
    predKNN = knn.predict(X_test)
    
    return predDT, predRFC, predlog, predSVC, predgnb, predKNN

# Create performance comparison chart
def create_performance_chart(y_test, predDT, predRFC, predlog, predSVC, predgnb, predKNN):
    chart = {
        'Metric': ["Accuracy", "F1-Score", "Recall", "Precision", "R2-Score"],
        'DT': [
            accuracy_score(y_test, predDT),
            f1_score(y_test, predDT),
            recall_score(y_test, predDT),
            precision_score(y_test, predDT),
            r2_score(y_test, predDT)
        ],
        'RF': [
            accuracy_score(y_test, predRFC),
            f1_score(y_test, predRFC),
            recall_score(y_test, predRFC),
            precision_score(y_test, predRFC),
            r2_score(y_test, predRFC)
        ],
        'LR': [
            accuracy_score(y_test, predlog),
            f1_score(y_test, predlog),
            recall_score(y_test, predlog),
            precision_score(y_test, predlog),
            r2_score(y_test, predlog)
        ],
        'SVM': [
            accuracy_score(y_test, predSVC),
            f1_score(y_test, predSVC),
            recall_score(y_test, predSVC),
            precision_score(y_test, predSVC),
            r2_score(y_test, predSVC)
        ],
        'NB': [
            accuracy_score(y_test, predgnb),
            f1_score(y_test, predgnb),
            recall_score(y_test, predgnb),
            precision_score(y_test, predgnb),
            r2_score(y_test, predgnb)
        ],
        'KNN': [
            accuracy_score(y_test, predKNN),
            f1_score(y_test, predKNN),
            recall_score(y_test, predKNN),
            precision_score(y_test, predKNN),
            r2_score(y_test, predKNN)
        ]
    }
    
    return pd.DataFrame(chart)

def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train models and get predictions
    predDT, predRFC, predlog, predSVC, predgnb, predKNN = train_and_predict(X_train, X_test, y_train)
    
    # Create performance comparison chart
    performance_chart = create_performance_chart(y_test, predDT, predRFC, predlog, predSVC, predgnb, predKNN)
    
    # Save chart to CSV
    performance_chart.to_csv('model_performance.csv', index=False)
    print("Model performance comparison saved to 'model_performance.csv'")
    
    return performance_chart

if __name__ == "__main__":
    chart = main()
    print(chart) 