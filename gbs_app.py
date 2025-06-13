import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv("GBS_Pune_Dataset.csv")

# Data preparation
df['Risk'] = df['Recovery_Status'].apply(lambda x: 1 if x == 'Hospitalized' else 0)
df['Onset_Month'] = pd.to_datetime(df['Onset_Date']).dt.month
X = df[['Age', 'Gender', 'Area', 'Onset_Month']]
y = df['Risk']

# Preprocessing
categorical_features = ['Gender', 'Area']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200))
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Streamlit UI
st.title("🧠 GBS Risk Prediction App")
st.markdown("Predict risk based on Age, Gender, Area, and Month of Onset.")

age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Select Gender", options=['M', 'F'])
area = st.selectbox("Select Area", options=df['Area'].unique())
onset_month = st.slider("Month of Onset", 1, 12, 6)

if st.button("Predict Risk"):
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Area': area,
        'Onset_Month': onset_month
    }])
    probability = model_pipeline.predict_proba(input_data)[0][1]
    st.success(f"Predicted GBS Risk: {round(probability * 100, 2)}%")
