import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load CSV
df = pd.read_csv("data/rice_leaf_data_hog_aug.csv")


X = df.drop("label", axis=1)
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
print("Training completed.")

# Test accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(clf, "artifacts/rf_model_hog.joblib")
print("Model saved at artifacts/rf_model_hog.joblib")
