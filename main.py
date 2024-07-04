import mysql.connector
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import os


# Fungsi untuk melakukan koneksi ke MySQL dan mendapatkan data dari tabel 'data_icd'
def get_data_icd_from_db():
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        # Query untuk mengambil semua data dari tabel 'data_icd'
        query = "SELECT * FROM data_icd"

        # Eksekusi query dan simpan hasilnya dalam DataFrame
        df_icd = pd.read_sql(query, con=db_connection)

        # Tutup koneksi database
        db_connection.close()

        return df_icd

    except mysql.connector.Error as err:
        print(f"Error while connecting to MySQL: {err}")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong jika terjadi kesalahan


# Fungsi untuk memprediksi label penyakit berdasarkan model dan TF-IDF vectorizer
def predict_label_disease(model, tfidf_vectorizer, disease_name):
    tfidf_text = tfidf_vectorizer.transform([disease_name])
    predicted_label = model.predict(tfidf_text)
    return predicted_label[0]


# Fungsi untuk mencari kandidat ICD berdasarkan query
def search_icd_candidates(query, dataset, model, tfidf_vectorizer):
    # Predict ICD category using the trained model
    predicted_label = predict_label_disease(model, tfidf_vectorizer, query)

    # Mapping of predicted labels to chapter ranges or codes
    chapter_mapping = {
        'Chapter I': 'A00-B99',
        'Chapter II': 'C00-D48',
        'Chapter III': 'D50-D89',
        'Chapter IV': 'E00-E90',
        'Chapter V': 'F00-F99',
        'Chapter VI': 'G00-G99',
        'Chapter VII': 'H00-H59',
        'Chapter VIII': 'H60-H95',
        'Chapter IX': 'I00-I99',
        'Chapter X': 'J00-J99',
        'Chapter XI': 'K00-K93',
        'Chapter XII': 'L00-L99',
        'Chapter XIII': 'M00-M99',
        'Chapter XIV': 'N00-N99',
        'Chapter XV': 'O00-O99',
        'Chapter XVI': 'P00-P96',
        'Chapter XVII': 'Q00-Q99',
        'Chapter XVIII': 'R00-R99',
        'Chapter XIX': 'S00-T98',
        'Chapter XX': 'V01-Y98',
        'Chapter XXI': 'Z00-Z99',
        'Chapter XXII': 'U00-U99'
    }

    # Get the range for the predicted chapter
    chapter_range = chapter_mapping.get(predicted_label)

    if not chapter_range:
        # If the predicted label is not in the mapping, return an empty DataFrame
        return pd.DataFrame()

    # Filter dataset based on the predicted chapter
    start_code, end_code = chapter_range.split('-')
    filtered_dataset = dataset[(dataset['kode_icd'] >= start_code) & (dataset['kode_icd'] <= end_code)]

    # Search for matching entries based on the query
    matching_entries = filtered_dataset[filtered_dataset['nama_penyakit'].str.contains(query, case=False)]

    return matching_entries


# Fungsi untuk menghapus data dari tabel 'search_icd'
def clear_search_icd_table():
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()

        # Hapus data dari tabel 'search_icd'
        delete_query = "DELETE FROM search_icd"
        cursor.execute(delete_query)

        # Commit transaksi
        db_connection.commit()

        # Tutup koneksi database
        cursor.close()
        db_connection.close()

    except mysql.connector.Error as err:
        print(f"Error while clearing MySQL table: {err}")


# Fungsi untuk menyisipkan hasil pencarian ke tabel 'search_icd'
def insert_search_results_to_db(results):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()

        # Query untuk menyisipkan data ke tabel 'search_icd'
        insert_query = "INSERT INTO search_icd (kode_icd, nama_penyakit) VALUES (%s, %s)"

        # Loop melalui hasil pencarian dan sisipkan satu per satu
        for result in results:
            cursor.execute(insert_query, (result['kode_icd'], result['nama_penyakit']))

        # Commit transaksi
        db_connection.commit()

        # Tutup koneksi database
        cursor.close()
        db_connection.close()

    except mysql.connector.Error as err:
        print(f"Error while inserting to MySQL: {err}")


# Inisialisasi Flask
app = Flask(__name__)


@app.route("/")
def main():
    return """
        Response Successful!
    """


# Load model dan TF-IDF vectorizer
model = joblib.load('model_svc.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')


# Definisikan route endpoint untuk menerima POST request
@app.route('/search_icd', methods=['POST'])
def search_icd():
    # Ambil data JSON dari request
    request_data = request.get_json()

    # Pastikan data JSON berisi 'query'
    if 'query' not in request_data:
        return jsonify({'error': 'Missing query parameter'}), 400

    # Ambil query dari data JSON
    query = request_data['query']

    # Panggil fungsi get_data_icd_from_db untuk mendapatkan data dari database
    df_icd = get_data_icd_from_db()

    # Panggil fungsi search_icd_candidates dengan query yang diberikan
    matching_entries = search_icd_candidates(query, df_icd, model, tfidf_vectorizer)

    # Hapus data lama dari tabel 'search_icd'
    clear_search_icd_table()

    # Format hasil pencarian sebagai JSON
    if not matching_entries.empty:
        result = matching_entries[['kode_icd', 'nama_penyakit']].to_dict(orient='records')

        # Sisipkan hasil pencarian ke tabel 'search_icd'
        insert_search_results_to_db(result)

        return jsonify(result)
    else:
        return jsonify({'message': f'No matching ICD codes found for query: {query}'})


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
