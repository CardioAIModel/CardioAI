# svm_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("cardio_train.csv", sep=';')  # assuming ; delimiter
print("Data loaded successfully")

# Separate features and target
X = df.drop(columns=["cardio"])
y = df["cardio"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")
