# Step 1 : Data Cleaning & Preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import pickle

# 1️ Load dataset
df = pd.read_csv("dataset.csv")

# 2️ Drop rows with missing symptoms or disease
df.dropna(subset=['Symptom_1', 'Disease'], inplace=True)

# 3️ Collect all symptom columns
symptom_cols = [col for col in df.columns if 'Symptom' in col]

# 4️ Merge all symptoms into a list per row
df['Symptom_List'] = df[symptom_cols].values.tolist()
df['Symptom_List'] = df['Symptom_List'].apply(lambda x: [i.strip() for i in x if isinstance(i, str)])

# Drop rows where no valid symptoms are left
df = df[df['Symptom_List'].map(len) > 0]

# 5️ Encode Symptoms with MultiLabelBinarizer ( Converts a list of symptoms into a format the model understands (0s and 1s))
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['Symptom_List'])

# 6️ Encode Diseases with LabelEncoder (Converts disease names (flu) into numbers (12))
le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# 7️ Save the encoders for later use in the app
with open("symptom_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f)
#(Saves the LabelEncoder for converting predicted numbers back to disease name)
with open("label_encoder.pkl", "wb") as f: 
    pickle.dump(le, f)

print("Step 1 Completed: Data cleaned and encoded!")
print(f"Total samples: {len(X)}, Total symptoms: {len(mlb.classes_)}, Total diseases: {len(le.classes_)}")

# Step 2 : 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 8️ Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9️ Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 10 Make predictions on test data
y_pred = model.predict(X_test)
  
# 11 Evaluate the model
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 12 Save the trained model 
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n Step 2 Completed: Model trained and saved as 'rf_model.pkl'")
