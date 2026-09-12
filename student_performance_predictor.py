import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("================================")
print(" STUDENT PERFORMANCE PREDICTOR")
print("================================")

# Dataset
data = {
    "study_hours": [1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11],
    "attendance": [60,65,70,68,75,72,78,80,82,85,84,88,90,91,93,94,95,96,97,98],
    "previous_score": [45,50,48,55,58,60,62,65,68,70,72,74,76,80,82,85,87,90,92,94],
    "final_score": [42,48,50,55,58,62,65,68,72,75,77,80,82,85,88,90,92,94,96,98]
}

df = pd.DataFrame(data)

# Features and target
X = df[["study_hours", "attendance", "previous_score"]]
y = df["final_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMODEL RESULTS")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# User input
print("\nNEW STUDENT PREDICTION")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))

new_student = pd.DataFrame(
    [[study_hours, attendance, previous_score]],
    columns=["study_hours", "attendance", "previous_score"]
)

result = model.predict(new_student)[0]

print("\nPredicted Final Score:", round(result, 2))

if result >= 90:
    level = "Excellent"
elif result >= 75:
    level = "Good"
elif result >= 50:
    level = "Average"
else:
    level = "Needs Improvement"

print("Performance Level:", level)

print("\nProject completed successfully!")
