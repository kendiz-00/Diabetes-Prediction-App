import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib

def create_model():
    # Load the dataset (replace with your actual dataset path)
    data = pd.read_csv('diabetes.csv')  # Ensure the CSV file is in the same directory or provide the full path

    # Display the first few rows to understand the dataset structure
    print(data.head())

    # Check for missing values
    print(data.isnull().sum())

    # Define features (X) and target (y)
    X = data.drop(columns=['Outcome'])  # 'Outcome' is the target column
    y = data['Outcome']

    # Convert categorical columns to numeric using one-hot encoding, if any (not needed here as all are numeric)
    X = pd.get_dummies(X)

    # Ensure only numeric data is passed to the scaler
    X = X.select_dtypes(include=['number'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train a Logistic Regression model
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the trained model and scaler
    joblib.dump(model, 'diabetes_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Model performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy:.2f}")

    # Print classification report and confusion matrix for more detailed evaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Call the function to create and test the model
if __name__ == '__main__':
    create_model()