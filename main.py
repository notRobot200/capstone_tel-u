from flask import Flask, request, jsonify
import joblib
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import mysql.connector

# Load model dan TF-IDF vectorizer
model = joblib.load('model_svm.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

Flask(__name__)

# Load model dan TF-IDF vectorizer
model = joblib.load('model_svm.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Koneksi ke MySQL
db_connection = mysql.connector.connect(
  host="34.145.29.172",
  user="beingman",
  password="123",
  database="oetomo"
)

app = Flask(__name__)

def predict_label_disease(disease_name):
    tfidf_text = tfidf_vectorizer.transform([disease_name])
    predicted_label = model.predict(tfidf_text)
    return predicted_label[0]

def search_icd_candidates(query, predicted_label):
    # Koneksi ke MySQL
    cursor = db_connection.cursor()

    # Query untuk mencari matching entries berdasarkan label yang diprediksi
    query = f"SELECT * FROM data_icd WHERE Kode_ICD LIKE '{predicted_label}%' AND Nama_Penyakit LIKE '%{query}%'"
    cursor.execute(query)
    matching_entries = cursor.fetchall()

    # Konversi hasil query menjadi DataFrame
    df_matching_entries = pd.DataFrame(matching_entries, columns=['Kode_ICD', 'Nama_Penyakit'])

    return df_matching_entries

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    disease_name = data['disease_name']
    predicted_label = predict_label_disease(disease_name)

    # Cari kandidat ICD berdasarkan nama penyakit dan label yang diprediksi
    matching_entries = search_icd_candidates(disease_name, predicted_label)

    return jsonify({'predicted_label': predicted_label, 'matching_entries': matching_entries.to_dict(orient='records')})

if __name__ == '__main__':
    app.run(debug=True)
