Disease-Prediction-Project
This project predicts diseases based on patient symptoms using a Random Forest Classifier. Symptoms are encoded using MultiLabelBinarizer and the model is deployed with Streamlit for real-time predictions. The system helps in quick preliminary diagnosis and can be extended for larger datasets.

Overview
This project predicts diseases based on patient symptoms using machine learning. A Random Forest Classifier is used for prediction, and a Streamlit app provides a simple interactive interface.

Features
Predicts 40+ diseases from input symptoms.
Encodes multiple symptoms using MultiLabelBinarizer.
Interactive Streamlit app for real-time predictions.

How to Use
1. Requirements : numpy, pandas, scikit-learn, streamlit, matplotlib, seaborn, joblib
2. Run the Streamlit app : streamlit run app.py

Tools & Libraries
- Python  
- Scikit-learn  
- Pandas, Numpy  
- Streamlit, Matplotlib, Seaborn  

Future Improvements
- Add more diseases and symptoms.  
- Improve model performance using hyperparameter tuning.  
- Deploy as a web app with database integration.

Project Structure:
data             # Dataset CSV files
model            # Saved trained model (.pkl)
src               # Python scripts
train_model.py      # Train the ML model
predict_disease.py   # Predict diseases from symptoms
utils.py             # Helper functions (optional)
app                   # Streamlit app (app.py)





