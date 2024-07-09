# precompute_vectors.py
import pandas as pd
import spacy
import mysql.connector

# Memuat model bahasa spaCy yang besar
nlp = spacy.load('en_core_web_lg')

# Fungsi untuk praproses teks
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

# Function to convert text to vector
def text_to_vector(text):
    return nlp(text).vector

def get_data_icd_from_db():
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        # Query untuk mengambil semua data dari tabel 'data_icd_en'
        query = "SELECT * FROM data_icd_en"

        # Eksekusi query dan simpan hasilnya dalam DataFrame
        df_icd = pd.read_sql(query, con=db_connection)

        # Tutup koneksi database
        db_connection.close()

        return df_icd

    except mysql.connector.Error as err:
        print(f"Error while connecting to MySQL: {err}")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong jika terjadi kesalahan

# Ambil data ICD dari database
df_icd = get_data_icd_from_db()

# Praproses dataset
df_icd['Nama_Penyakit_Processed'] = df_icd['nama_penyakit'].apply(preprocess_text)

# Precompute vectors for dataset
df_icd['Nama_Penyakit_Vector'] = df_icd['Nama_Penyakit_Processed'].apply(text_to_vector)

# Simpan hasil ke file untuk digunakan nanti
df_icd.to_pickle('precomputed_icd_vectors.pkl')