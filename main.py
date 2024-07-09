import mysql.connector
import pandas as pd
from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS  # Import CORS
from googletrans import Translator
import spacy
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  # Enable CORS

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

# Fungsi untuk mengambil hasil pencarian dari tabel 'search_icd'
def get_search_icd_results_from_db():
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        # Query untuk mengambil semua data dari tabel 'search_icd'
        query = "SELECT * FROM search_icd"

        # Eksekusi query dan simpan hasilnya dalam DataFrame
        df_search_icd = pd.read_sql(query, con=db_connection)

        # Tutup koneksi database
        db_connection.close()

        return df_search_icd

    except mysql.connector.Error as err:
        print(f"Error while connecting to MySQL: {err}")
        return pd.DataFrame()  # Mengembalikan DataFrame kosong jika terjadi kesalahan


# Fungsi untuk memprediksi label penyakit berdasarkan model dan TF-IDF vectorizer
def predict_label_disease(model, tfidf_vectorizer, disease_name):
    tfidf_text = tfidf_vectorizer.transform([disease_name])
    predicted_label = model.predict(tfidf_text)
    return predicted_label[0]


# # Fungsi untuk mencari kandidat ICD berdasarkan query
# def search_icd_candidates(query, dataset, model, tfidf_vectorizer):
#     # Predict ICD category using the trained model
#     predicted_label = predict_label_disease(model, tfidf_vectorizer, query)
#
#     # Mapping of predicted labels to chapter ranges or codes
#     chapter_mapping = {
#         'Chapter I': 'A00-B99',
#         'Chapter II': 'C00-D48',
#         'Chapter III': 'D50-D89',
#         'Chapter IV': 'E00-E90',
#         'Chapter V': 'F00-F99',
#         'Chapter VI': 'G00-G99',
#         'Chapter VII': 'H00-H59',
#         'Chapter VIII': 'H60-H95',
#         'Chapter IX': 'I00-I99',
#         'Chapter X': 'J00-J99',
#         'Chapter XI': 'K00-K93',
#         'Chapter XII': 'L00-L99',
#         'Chapter XIII': 'M00-M99',
#         'Chapter XIV': 'N00-N99',
#         'Chapter XV': 'O00-O99',
#         'Chapter XVI': 'P00-P96',
#         'Chapter XVII': 'Q00-Q99',
#         'Chapter XVIII': 'R00-R99',
#         'Chapter XIX': 'S00-T98',
#         'Chapter XX': 'V01-Y98',
#         'Chapter XXI': 'Z00-Z99',
#         'Chapter XXII': 'U00-U99'
#     }
#
#     # Get the range for the predicted chapter
#     chapter_range = chapter_mapping.get(predicted_label)
#
#     if not chapter_range:
#         # If the predicted label is not in the mapping, return an empty DataFrame
#         return pd.DataFrame()
#
#     # Filter dataset based on the predicted chapter
#     start_code, end_code = chapter_range.split('-')
#     filtered_dataset = dataset[(dataset['kode_icd'] >= start_code) & (dataset['kode_icd'] <= end_code)]
#
#     # Search for matching entries based on the query
#     matching_entries = filtered_dataset[filtered_dataset['nama_penyakit'].str.contains(query, case=False)]
#
#     return matching_entries


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


def insert_selected_icd_to_db(kode_icd, nama_penyakit):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()

        # Query untuk menyisipkan data ke tabel 'selected_icd'
        insert_query = "INSERT INTO selected_icd (kode_icd, nama_penyakit) VALUES (%s, %s)"
        cursor.execute(insert_query, (kode_icd, nama_penyakit))

        # Commit transaksi
        db_connection.commit()

        # Tutup koneksi database
        cursor.close()
        db_connection.close()

    except mysql.connector.Error as err:
        print(f"Error while inserting to MySQL: {err}")

def insert_query_to_db(query):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()

        # Query untuk menyisipkan data ke tabel 'query_disease'
        insert_query = "INSERT INTO query_disease (query) VALUES (%s)"
        cursor.execute(insert_query, (query,))

        # Commit transaksi
        db_connection.commit()

        # Tutup koneksi database
        cursor.close()
        db_connection.close()

    except mysql.connector.Error as err:
        print(f"Error while inserting to MySQL: {err}")

#MASIH PENGEMBANGAN
# # Database connection
# def get_db_connection():
#     return mysql.connector.connect(
#         host="34.145.29.172",
#         user="beingman",
#         password="123",
#         database="oetomo"
#     )
#
# # Function to fetch data from a table
# def fetch_data_from_table(table_name):
#     db_connection = get_db_connection()
#     query = f"SELECT * FROM `{table_name}`"
#     df = pd.read_sql(query, db_connection)
#     db_connection.close()
#     return df
#
# # Function to fetch the disease name from query_disease table
# def fetch_disease_name():
#     db_connection = get_db_connection()
#     query = "SELECT query FROM query_disease ORDER BY id DESC LIMIT 1;"  # Adjust query as needed
#     cursor = db_connection.cursor()
#     cursor.execute(query)
#     result = cursor.fetchone()
#     db_connection.close()
#     return result[0] if result else None
#
# # Search functions
# def search_disease(dataset, keywords, sort_column='TARIF KELAS 1', ascending=False):
#     if 'TARIF KELAS' in sort_column:
#         matching_entries = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(keywords, case=False)]
#     else:
#         matching_entries = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(keywords, case=False)].sort_values(by=sort_column, ascending=ascending)
#     return matching_entries
#
# def search_disease_keywords(dataset, keywords, sort_column='TARIF KELAS 1', ascending=False):
#     all_matches = pd.DataFrame()
#     for key in keywords:
#         matches = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(key, case=False)]
#         all_matches = pd.concat([all_matches, matches])
#     return all_matches.sort_values(by=sort_column, ascending=ascending)
# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="34.145.29.172",
        user="beingman",
        password="123",
        database="oetomo"
    )

# Function to fetch data from a table
def fetch_data_from_table(table_name):
    db_connection = get_db_connection()
    query = f"SELECT * FROM `{table_name}`"
    df = pd.read_sql(query, db_connection)
    db_connection.close()
    return df

# Function to fetch the disease name from query_disease table
def fetch_disease_name():
    db_connection = get_db_connection()
    query = "SELECT query FROM query_disease ORDER BY id DESC LIMIT 1;"
    cursor = db_connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    db_connection.close()
    return result[0] if result else None

# Search functions
def search_disease(dataset, keywords, sort_column='TARIF KELAS 1', ascending=False):
    if 'TARIF KELAS' in sort_column:
        matching_entries = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(keywords, case=False)]
    else:
        matching_entries = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(keywords, case=False)].sort_values(by=sort_column, ascending=ascending)
    return matching_entries

def search_disease_keywords(dataset, keywords, sort_column='TARIF KELAS 1', ascending=False):
    all_matches = pd.DataFrame()
    for key in keywords:
        matches = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(key, case=False)]
        all_matches = pd.concat([all_matches, matches])
    return all_matches.sort_values(by=sort_column, ascending=ascending)
# #MASIH PENGEMBANGAN

# Fungsi untuk menerjemahkan teks ke bahasa Inggris
def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='id', dest='en')
    return translation.text

# Fungsi untuk praproses teks
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

# Function to convert text to vector
def text_to_vector(text):
    return nlp(text).vector

# Fungsi untuk mencari kandidat ICD menggunakan NLP
def search_icd_candidates(query, dataset):
    # Praproses query
    query_processed = preprocess_text(query)
    query_vector = nlp(query_processed).vector

    # Calculate similarity scores
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)

    dataset['similarity'] = dataset['Nama_Penyakit_Vector'].apply(lambda x: cosine_similarity(x, query_vector))

    # Urutkan berdasarkan skor kemiripan dan kembalikan hasil teratas
    matching_entries = dataset.sort_values(by='similarity', ascending=False).head(10)

    return matching_entries

# Load model dan TF-IDF vectorizer
model = joblib.load('model_svc_en.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_en.pkl')
# Memuat model bahasa spaCy yang besar
nlp = spacy.load('en_core_web_lg')

@app.route("/")
def main():
    return """
        Response Successful!
    """

@app.route('/search_icd', methods=['POST'])
def search_icd():
    # Ambil data JSON dari request
    request_data = request.get_json()

    # Pastikan data JSON berisi 'query'
    if 'query' not in request_data:
        return jsonify({'error': 'Missing query parameter'}), 400

    # Ambil query dari data JSON
    query = request_data['query']
    use_nlp = request_data.get('use_nlp', False)  # Default to False if not provided

    # Terjemahkan query ke bahasa Inggris
    query_in_english = translate_to_english(query)

    if use_nlp:
        # Panggil fungsi search_icd_candidates dengan query yang diberikan
        matching_entries = search_icd_candidates(query_in_english, get_data_icd_from_db())
        # Format hasil pencarian sebagai JSON
        if not matching_entries.empty:
            result = matching_entries[['kode_icd', 'nama_penyakit', 'similarity']].to_dict(orient='records')
            return jsonify(result)
        else:
            return jsonify({'message': f'No matching ICD codes found for query: {query_in_english}'})
    else:
        # Sisipkan query ke dalam tabel 'query_disease'
        insert_query_to_db(query)

        # Panggil fungsi get_data_icd_from_db untuk mendapatkan data dari database
        df_icd_manual = get_data_icd_from_db()

        # Hapus data lama dari tabel 'search_icd'
        clear_search_icd_table()

        # Filter data ICD berdasarkan query manual
        matching_entries = df_icd_manual[df_icd_manual['nama_penyakit'].str.contains(query_in_english, case=False, na=False)]

        # Format hasil pencarian sebagai JSON
        if not matching_entries.empty:
            result = matching_entries[['kode_icd', 'nama_penyakit']].to_dict(orient='records')

            # Sisipkan hasil pencarian ke tabel 'search_icd'
            insert_search_results_to_db(result)

            return jsonify(result)
        else:
            return jsonify({'message': f'No matching ICD codes found for query: {query}'})


@app.route('/select_icd', methods=['POST'])
def select_icd():
    # Ambil data JSON dari request
    request_data = request.get_json()

    # Pastikan data JSON berisi 'index'
    if 'index' not in request_data:
        return jsonify({'error': 'Missing index parameter'}), 400

    # Ambil index dari data JSON
    selected_index = request_data['index'] - 1

    # Panggil fungsi get_search_icd_results_from_db untuk mendapatkan hasil pencarian dari database
    df_search_icd = get_search_icd_results_from_db()

    # Periksa apakah index valid
    if selected_index < 0 or selected_index >= len(df_search_icd):
        return jsonify({'error': 'Invalid index parameter'}), 400

    # Memilih hasil pencarian berdasarkan indeks yang dimasukkan pengguna
    selected_entry = df_search_icd.iloc[selected_index]

    # Simpan Kode_ICD dan Nama_Penyakit ke dalam variabel
    selected_kode_icd = selected_entry['kode_icd']
    selected_nama_penyakit = selected_entry['nama_penyakit']

    # Sisipkan ke tabel 'selected_icd'
    insert_selected_icd_to_db(selected_kode_icd, selected_nama_penyakit)

    # Tampilkan informasi yang dipilih
    return jsonify({
        'selected_kode_icd': selected_kode_icd,
        'selected_nama_penyakit': selected_nama_penyakit
    })

#MASIH PENGEMBANGAN
# # API endpoint for searching diseases
# @app.route('/search_disease', methods=['GET'])
# def search_disease_api():
#     # Input parameters
#     jenis_pelayanan = request.args.get('jenis_pelayanan', default='inap').strip().lower()
#
#     # Fetch disease name from the database
#     disease_name = fetch_disease_name()
#     if not disease_name:
#         return jsonify({"error": "Tidak ada nama penyakit yang ditemukan di database."}), 400
#
#     # Selecting dataset based on input
#     if jenis_pelayanan == 'inap':
#         dataset = fetch_data_from_table('data_ina-cbg')
#     elif jenis_pelayanan == 'jalan':
#         dataset = fetch_data_from_table('df_ina_rj')
#     else:
#         return jsonify({"error": "Jenis pelayanan yang dimasukkan tidak valid."}), 400
#
#     # Sorting column based on input
#     sort_column_input = request.args.get('sort_column', default='TARIF KELAS 1').strip()
#     ascending_input = request.args.get('ascending', default='n').strip().lower()
#     ascending = ascending_input == 'y' if ascending_input in ['y', 'n'] else False
#
#     if dataset is fetch_data_from_table('df_ina_rj'):
#         sort_column = 'TARIF INA-CBG'
#     else:
#         if sort_column_input.isdigit():
#             if sort_column_input in ['1', '2', '3']:
#                 sort_column = 'TARIF KELAS ' + sort_column_input
#             else:
#                 return jsonify({"error": "Nomor kolom tarif kelas harus '1', '2', atau '3'."}), 400
#         elif sort_column_input.startswith('TARIF KELAS') and sort_column_input.split()[2].isdigit():
#             sort_column = sort_column_input
#         else:
#             sort_column = 'TARIF KELAS 1'
#
#     # Searching disease in the selected dataset
#     matching_entries = search_disease(dataset, disease_name, sort_column=sort_column, ascending=ascending)
#
#     if matching_entries.empty:
#         # If no results match, search based on keywords
#         keywords = disease_name.split()
#         matching_entries = search_disease_keywords(dataset, keywords, sort_column=sort_column, ascending=ascending)
#
#     # Formatting the results
#     if not matching_entries.empty:
#         if dataset is fetch_data_from_table('df_ina_rj'):
#             result = matching_entries[['KODE INA-CBG', 'DESKRIPSI KODE INA-CBG', 'TARIF INA-CBG']].head(10).to_dict(orient='records')
#         else:
#             tarif_columns = [col for col in matching_entries.columns if 'TARIF KELAS' in col]
#             if tarif_columns:
#                 result = matching_entries[['KODE INA-CBG', 'DESKRIPSI KODE INA-CBG', sort_column]].head(10).to_dict(orient='records')
#             else:
#                 result = "Tidak ada kolom tarif kelas yang cocok untuk ditampilkan."
#     else:
#         result = "Tidak ada hasil yang cocok."
#
#     return jsonify(result)
# API endpoint for searching diseases
@app.route('/search_disease', methods=['GET'])
def search_disease_api():
    # Input parameters
    jenis_pelayanan = request.args.get('jenis_pelayanan', default='inap').strip().lower()

    # Fetch disease name from the database
    disease_name = fetch_disease_name()
    if not disease_name:
        return jsonify({"error": "Tidak ada nama penyakit yang ditemukan di database."}), 400

    # Selecting dataset based on input
    if jenis_pelayanan == 'inap':
        dataset = fetch_data_from_table('data_ina-cbg')
        default_sort_column = 'TARIF KELAS 1'
    elif jenis_pelayanan == 'jalan':
        dataset = fetch_data_from_table('df_ina_rj')
        default_sort_column = 'TARIF INA-CBG'
    else:
        return jsonify({"error": "Jenis pelayanan yang dimasukkan tidak valid."}), 400

    # Sorting column based on input
    sort_column_input = request.args.get('sort_column', default=default_sort_column).strip()
    ascending_input = request.args.get('ascending', default='n').strip().lower()
    ascending = ascending_input == 'y' if ascending_input in ['y', 'n'] else False

    if jenis_pelayanan == 'jalan':
        sort_column = 'TARIF INA-CBG'
    else:
        if sort_column_input.isdigit():
            if sort_column_input in ['1', '2', '3']:
                sort_column = 'TARIF KELAS ' + sort_column_input
            else:
                return jsonify({"error": "Nomor kolom tarif kelas harus '1', '2', atau '3'."}), 400
        elif sort_column_input.startswith('TARIF KELAS') and sort_column_input.split()[2].isdigit():
            sort_column = sort_column_input
        else:
            sort_column = default_sort_column

    # Searching disease in the selected dataset
    matching_entries = search_disease(dataset, disease_name, sort_column=sort_column, ascending=ascending)

    if matching_entries.empty:
        # If no results match, search based on keywords
        keywords = disease_name.split()
        matching_entries = search_disease_keywords(dataset, keywords, sort_column=sort_column, ascending=ascending)

    # Formatting the results
    if not matching_entries.empty:
        if jenis_pelayanan == 'jalan':
            result = matching_entries[['KODE INA-CBG', 'DESKRIPSI KODE INA-CBG', 'TARIF INA-CBG']].head(10).to_dict(orient='records')
        else:
            tarif_columns = [col for col in matching_entries.columns if 'TARIF KELAS' in col]
            if tarif_columns:
                result = matching_entries[['KODE INA-CBG', 'DESKRIPSI KODE INA-CBG', sort_column]].head(10).to_dict(orient='records')
            else:
                result = "Tidak ada kolom tarif kelas yang cocok untuk ditampilkan."
    else:
        result = "Tidak ada hasil yang cocok."

    return jsonify(result)
#MASIH PENGEMBANGAN

if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
