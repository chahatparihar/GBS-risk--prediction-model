import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
data = {
    "Age": [25, 40, 55, 30, 60, 35, 45, 50, 29, 70],
    "Gender": [1, 0, 1, 0, 0, 1, 1, 0, 1, 0],  # 1 = Male, 0 = Female
    "Area": [1, 2, 1, 2, 3, 1, 3, 2, 1, 3],    # Encoded area
    "Date_Onset_DaysAgo": [5, 15, 20, 8, 12, 18, 7, 30, 3, 22],
    "Recovered": [1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    "High_Risk": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # Target variable
}

df = pd.DataFrame(data)

# Feature-target split
X = df.drop("High_Risk", axis=1)
y = df["High_Risk"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get user input
age = int(input("Enter Age: "))
gender = int(input("Enter Gender (1 = Male, 0 = Female): "))
area = int(input("Enter Area (1, 2, or 3): "))
onset_days = int(input("Enter Days since onset of symptoms: "))
recovered = int(input("Enter Recovery status (1 = Yes, 0 = No): "))

# Create user input data in same format as training data
user_input = [[age, gender, area, onset_days, recovered]]

# Predict probability
y_prob = model.predict_proba(user_input)  # user_input = user ke input ke values

# Display probability for each class
prob_high_risk = y_prob[0][1] * 100  # Probability for class 1 (High Risk)
prob_low_risk = y_prob[0][0] * 100   # Probability for class 0 (Low Risk)

# Print the percentage chances
print(f"Chances of High Risk (GBS) = {prob_high_risk:.2f}%")
print(f"Chances of Low Risk = {prob_low_risk:.2f}%")



