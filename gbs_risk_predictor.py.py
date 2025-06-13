import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv("GBS_Pune_Dataset.csv")  # Replace with actual file path

# Prepare data
df['Risk'] = df['Recovery_Status'].apply(lambda x: 1 if x == 'Hospitalized' else 0)
df['Onset_Month'] = pd.to_datetime(df['Onset_Date']).dt.month
X = df[['Age', 'Gender', 'Area', 'Onset_Month']]
y = df['Risk']

# Preprocessing pipeline
categorical_features = ['Gender', 'Area']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Build model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Function to predict GBS risk
def predict_gbs_risk(age, gender, area, onset_month):
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Area': area,
        'Onset_Month': onset_month
    }])
    probability = model_pipeline.predict_proba(input_data)[0][1]
    return round(probability * 100, 2)

# Command line interaction
def run_gbs_risk_model():
    print("\n📊 GBS Risk Prediction Model")
    print("-------------------------------")
    try:
        age = int(input("Enter Age: "))
        gender = input("Enter Gender (M/F): ").strip().upper()
        area = input("Enter Area (e.g., Viman Nagar, Sinhagad Rd): ").strip()
        onset_month = int(input("Enter Month of Onset (1-12): "))
        risk = predict_gbs_risk(age, gender, area, onset_month)
        print(f"\n🧠 Predicted GBS Risk: {risk}%")
    except Exception as e:
        print(f"⚠️ Error: {e}")

# Run the model
run_gbs_risk_model()



