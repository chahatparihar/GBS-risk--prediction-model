# GBS Risk Prediction Model 

This project is a machine learning-based solution designed to predict the risk of **Guillain–Barré Syndrome (GBS)** using healthcare-related data. The goal is to assist in early detection and response, especially during outbreak situations.

 Project Structure

 GBS-risk--prediction-model
├── data/
│ └── gbs_dataset.csv # Cleaned dataset used for training
├── notebooks/
│ └── gbs_model_training.ipynb # Model building and evaluation
├── models/
│ └── gbs_model.pkl # Saved ML model
├── requirements.txt # Python dependencies
└── main.py # Script to make predictions



 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook
- Flask / Streamlit (for deployment)

 Features

- Data preprocessing and cleaning
- Feature selection and scaling
- ML Model training (Logistic Regression / Random Forest / XGBoost)
- Evaluation metrics (Accuracy, ROC-AUC, etc.)
- Simple UI for prediction (optional)

 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/chahatparihar/GBS-risk--prediction-model.git
cd GBS-risk--prediction-model
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the notebook
Open the Jupyter Notebook to explore the model:

bash
Copy
Edit
jupyter notebook notebooks/gbs_model_training.ipynb
4. Run prediction script
bash
Copy
Edit
python main.py
 Results
Model Accuracy: ~92%

ROC AUC Score: 0.89

Confusion Matrix and Classification Report included in notebook

 Dataset Source
The dataset used is based on simulated or publicly available healthcare data and does not violate patient confidentiality.

 Author
Chahat Parihar
 LinkedIn

 Contribute
Feel free to fork this project, make improvements, and open a PR!

 License
This project is licensed under the MIT License. See LICENSE file for details.

yaml
Copy
Edit

---

Let me know if you want a lighter or more detailed version, or if you're using Streamlit/Flask UI – I can tailor it accordingly.








