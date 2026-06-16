import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
data = pd.read_csv("heart.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", data.shape)

# Convert all text columns to numeric values
for col in data.columns:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# Target Column
target_column = "num"

# Features and Target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

print("\n===== MODEL COMPARISON =====")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name}: {accuracy * 100:.2f}%")

# Find Best Model
best_model = max(results, key=results.get)

print("\n==========================")
print("Best Model:", best_model)
print("Best Accuracy:", round(results[best_model] * 100, 2), "%")
print("==========================")

# Accuracy Graph
plt.figure(figsize=(8, 5))

plt.bar(
    results.keys(),
    [score * 100 for score in results.values()]
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy (%)")

plt.tight_layout()

plt.show()