import mysql.connector
import pandas as pd
from flask import Flask, request, jsonify, Response
import joblib
import os
from flask_cors import CORS
from googletrans import Translator
import spacy
import numpy as np
from mysql.connector import OperationalError, Error
import nltk
from nltk.corpus import stopwords
import threading
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Adjust CORS settings as necessary
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
lock = threading.Lock()

df_icd = pd.read_pickle('precomputed_icd_vectors.pkl')
model = joblib.load('model_svc_en.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer_en.pkl')
nlp = spacy.load('en_core_web_lg')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

@app.route("/")
def main():
    return """
        Response Successful!
    """

@app.route('/search_icd_manual', methods=['POST'])
def search_icd_manual():
    request_data = request.get_json()

    if 'query' not in request_data or 'user_id' not in request_data:
        return jsonify({'error': 'Missing query or user_id parameter'}), 400

    query = request_data['query']
    user_id = request_data['user_id']
    nama_pasien = request_data['nama_pasien']

    query_in_english = translate_to_english(query)
    insert_query_to_db(user_id, query, nama_pasien)
    clear_search_icd_table(user_id)

    matching_entries_manual = search_icd_candidates_manual(query_in_english, get_data_icd_from_db(), model,
                                                           tfidf_vectorizer)

    if not matching_entries_manual.empty:
        matching_entries_manual['similarity'] = 1.0
        result_manual = matching_entries_manual[['kode_icd', 'nama_penyakit', 'similarity']].to_dict(orient='records')
        with lock:
            insert_search_results_to_db(user_id, result_manual)
        return jsonify(result_manual)
    else:
        result_manual = [
            {
                'kode_icd': "Maaf, kode tidak dapat ditemukan",
                'nama_penyakit': "Maaf, nama penyakit tidak dapat ditemukan",
                'similarity': 0.0
            }
        ]
        with lock:
            insert_search_results_to_db(user_id, result_manual)
        return jsonify(result_manual)


@app.route('/search_icd_nlp', methods=['POST'])
def search_icd_nlp():
    request_data = request.get_json()

    if 'query' not in request_data or 'user_id' not in request_data:
        return jsonify({'error': 'Missing query or user_id parameter'}), 400

    query = request_data['query']
    user_id = request_data['user_id']

    query_in_english = translate_to_english(query)
    # insert_query_to_db(user_id, query)
    clear_search_icd_table(user_id)

    matching_entries_nlp = search_icd_candidates(query_in_english, df_icd)

    if not matching_entries_nlp.empty:
        result_nlp = matching_entries_nlp[['kode_icd', 'nama_penyakit', 'similarity']].to_dict(orient='records')
        insert_search_results_to_db(user_id, result_nlp)
        return jsonify(result_nlp)
    else:
        result_nlp = [
            {
                'kode_icd': "Maaf, kode tidak dapat ditemukan",
                'nama_penyakit': "Maaf, nama penyakit tidak dapat ditemukan",
                'similarity': 0.0
            }
        ]
        with lock:
            insert_search_results_to_db(user_id, result_nlp)
        return jsonify(result_nlp)

@app.route('/search_icd_ctrlf', methods=['POST'])
def search_icd_ctrlf():
    request_data = request.get_json()

    if 'query' not in request_data or 'user_id' not in request_data:
        return jsonify({'error': 'Missing query or user_id parameter'}), 400

    query = request_data['query']
    user_id = request_data['user_id']

    query_in_english = translate_to_english(query)
    # insert_query_to_db(user_id, query)
    clear_search_icd_table(user_id)

    # Replace the old function with the new one
    matching_entries_ctrlf = search_icd_candidates_ctrlf(get_data_icd_from_db(), query_in_english)

    if not matching_entries_ctrlf.empty:
        matching_entries_ctrlf['similarity'] = 0.99
        result_ctrlf = matching_entries_ctrlf[['kode_icd', 'nama_penyakit', 'similarity']].to_dict(orient='records')
        with lock:
            insert_search_results_to_db(user_id, result_ctrlf)
        return jsonify(result_ctrlf)
    else:
        result_ctrlf = [
            {
                'kode_icd': "Maaf, kode tidak dapat ditemukan",
                'nama_penyakit': "Maaf, nama penyakit tidak dapat ditemukan",
                'similarity': 0.0
            }
        ]
        with lock:
            insert_search_results_to_db(user_id, result_ctrlf)
        return jsonify(result_ctrlf)


def search_icd_candidates_ctrlf(df, search_term):
    # Cari kecocokan langsung dengan kalimat pencarian
    result = df[df['nama_penyakit'].str.contains(search_term, case=False, na=False)]

    # Jika tidak ada kecocokan langsung, cari untuk setiap kata penting dalam kalimat pencarian
    if result.empty:
        words = search_term.split()
        important_words = [word for word in words if word.lower() not in stop_words]

        for word in important_words:
            word_result = df[df['nama_penyakit'].str.contains(word, case=False, na=False)]
            result = pd.concat([result, word_result]).drop_duplicates()

    return result

def get_data_icd_from_db():
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        query = "SELECT * FROM data_icd_en"

        df_icd = pd.read_sql(query, con=db_connection)

        db_connection.close()

        return df_icd

    except mysql.connector.Error as err:
        print(f"Error while connecting to MySQL: {err}")
        return pd.DataFrame()

# Fungsi untuk praproses teks
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])

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

# Fungsi untuk memprediksi label penyakit berdasarkan model dan TF-IDF vectorizer
def predict_label_disease(model, tfidf_vectorizer, disease_name):
    tfidf_text = tfidf_vectorizer.transform([disease_name])
    predicted_label = model.predict(tfidf_text)
    return predicted_label[0]


def search_icd_candidates_manual(query, dataset, model, tfidf_vectorizer):

    predicted_label = predict_label_disease(model, tfidf_vectorizer, query)

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

    chapter_range = chapter_mapping.get(predicted_label)

    if not chapter_range:
        return pd.DataFrame()

    filtered_dataset = dataset[dataset['kategori'] == predicted_label].copy()
    matching_entries = filtered_dataset[filtered_dataset['nama_penyakit'].str.contains(query, case=False)]

    return matching_entries

def insert_query_to_db(user_id, query, nama_pasien):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()

        insert_query = "INSERT INTO query_disease (user_id, query, patient_name) VALUES (%s, %s, %s)"
        cursor.execute(insert_query, (user_id, query, nama_pasien))

        db_connection.commit()

        cursor.close()
        db_connection.close()

        # Emit event to update patient names
        socketio.emit('update_patient_names', {'user_id': user_id})

    except mysql.connector.Error as err:
        print(f"Error while inserting to MySQL: {err}")

def insert_search_results_to_db(user_id, results):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()
        insert_query = "INSERT INTO search_icd (user_id, kode_icd, nama_penyakit, similarity) VALUES (%s, %s, %s, %s)"

        for result in results:
            cursor.execute(insert_query, (user_id, result['kode_icd'], result['nama_penyakit'], result['similarity']))

        db_connection.commit()

        cursor.close()
        db_connection.close()

    except mysql.connector.Error as err:
        print(f"Error while inserting to MySQL: {err}")

def clear_search_icd_table(user_id):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()

        delete_query = "DELETE FROM search_icd WHERE user_id = %s"
        cursor.execute(delete_query, (user_id,))

        db_connection.commit()

        cursor.close()
        db_connection.close()

    except mysql.connector.Error as err:
        print(f"Error while clearing MySQL table: {err}")

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='id', dest='en')
    return translation.text


@app.route('/select_icd', methods=['POST'])
def select_icd():
    request_data = request.get_json()

    if 'nama_penyakit' not in request_data or 'user_id' not in request_data or 'nama_pasien' not in request_data or 'query' not in request_data:
        return jsonify({'error': 'Missing nama_penyakit, user_id, nama_pasien, or query parameter'}), 400

    nama_penyakit = request_data['nama_penyakit']
    user_id = request_data['user_id']
    nama_pasien = request_data['nama_pasien']
    query_param = request_data['query']

    df_search_icd = get_search_icd_results_from_db(user_id)

    selected_entries = df_search_icd[df_search_icd['nama_penyakit'] == nama_penyakit]

    if selected_entries.empty:
        return jsonify({'error': 'Invalid nama_penyakit parameter'}), 400

    selected_entry = selected_entries.iloc[0]  # assuming you want the first match if there are multiple

    selected_kode_icd = selected_entry['kode_icd']
    selected_nama_penyakit = selected_entry['nama_penyakit']

    insert_selected_icd_to_db(user_id, selected_kode_icd, selected_nama_penyakit, nama_pasien, query_param)

    return jsonify({
        'selected_kode_icd': selected_kode_icd,
        'selected_nama_penyakit': selected_nama_penyakit
    })

def get_search_icd_results_from_db(user_id):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        query = "SELECT kode_icd, nama_penyakit, similarity FROM search_icd WHERE user_id = %s"
        df_search_icd = pd.read_sql(query, db_connection, params=(user_id,))

        db_connection.close()
        return df_search_icd

    except mysql.connector.Error as err:
        print(f"Error while fetching search results from MySQL: {err}")
        return pd.DataFrame()

def insert_selected_icd_to_db(user_id, kode_icd, nama_penyakit, nama_pasien, query):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()

        # Query untuk menyisipkan data ke tabel 'selected_icd'
        insert_query = "INSERT INTO selected_icd (user_id, kode_icd, nama_penyakit, patient_name, query) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(insert_query, (user_id, kode_icd, nama_penyakit, nama_pasien, query))

        # Commit transaksi
        db_connection.commit()

        # Tutup koneksi database
        cursor.close()
        db_connection.close()

    except mysql.connector.Error as err:
        print(f"Error while inserting selected ICD to MySQL: {err}")

@app.route('/search_disease', methods=['GET'])
def search_disease_api():
    jenis_pelayanan = request.args.get('jenis_pelayanan', default='inap').strip().lower()
    user_id = request.args.get('user_id')
    query_term = request.args.get('query')

    if not user_id:
        return jsonify({"error": "User ID tidak ditemukan."}), 400

    disease_names = fetch_disease_names_by_user_id(user_id)
    if not disease_names:
        return jsonify({"error": "Tidak ada nama penyakit yang ditemukan di database untuk user ID tersebut."}), 400

    if query_term not in disease_names:
        return jsonify({"error": "Query term tidak ditemukan untuk user ID tersebut."}), 400

    disease_name = query_term
    print(f"Searching for disease name '{disease_name}' for user_id {user_id}")  # Debugging line

    if jenis_pelayanan == 'inap':
        dataset = fetch_data_from_table('data_ina-cbg')
        default_sort_column = 'TARIF KELAS 1'
    elif jenis_pelayanan == 'jalan':
        dataset = fetch_data_from_table('df_ina_rj')
        default_sort_column = 'TARIF INA-CBG'
    else:
        return jsonify({"error": "Jenis pelayanan yang dimasukkan tidak valid."}), 400

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

    matching_entries = search_disease(dataset, disease_name, sort_column=sort_column, ascending=ascending)

    if matching_entries.empty:
        keywords = disease_name.split()
        matching_entries = search_disease_keywords(dataset, keywords, sort_column=sort_column, ascending=ascending)

    if not matching_entries.empty:
        if jenis_pelayanan == 'jalan':
            result = matching_entries[['KODE INA-CBG', 'DESKRIPSI KODE INA-CBG', 'TARIF INA-CBG']].head(10).to_dict(
                orient='records')
        else:
            tarif_columns = [col for col in matching_entries.columns if 'TARIF KELAS' in col]
            if tarif_columns:
                result = matching_entries[['KODE INA-CBG', 'DESKRIPSI KODE INA-CBG', sort_column]].head(10).to_dict(
                    orient='records')
            else:
                result = "Tidak ada kolom tarif kelas yang cocok untuk ditampilkan."
    else:
        result = "Tidak ada hasil yang cocok."

    return jsonify(result)

def fetch_disease_names_by_user_id(user_id):
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )

        cursor = db_connection.cursor()
        query = "SELECT query FROM query_disease WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        result = cursor.fetchall()

        cursor.close()
        db_connection.close()

        if result:
            queries = [row[0] for row in result]
            print(f"Fetched queries for user_id {user_id}: {queries}")  # Debugging line
            return queries
        else:
            return None

    except mysql.connector.Error as err:
        print(f"Error while fetching disease names from MySQL: {err}")
        return None


def get_db_connection():
    return mysql.connector.connect(
        host="34.145.29.172",
        user="beingman",
        password="123",
        database="oetomo"
    )

def fetch_data_from_table(table_name):
    db_connection = get_db_connection()
    query = f"SELECT * FROM `{table_name}`"
    df = pd.read_sql(query, db_connection)
    db_connection.close()
    return df

# Search functions
def search_disease(dataset, keywords, sort_column='TARIF KELAS 1', ascending=False):
    if 'TARIF KELAS' in sort_column:
        matching_entries = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(keywords, case=False)]
    else:
        matching_entries = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(keywords, case=False)].sort_values(by=sort_column, ascending=ascending)
    print(f"Matching entries for keywords '{keywords}': {matching_entries}")  # Debugging line
    return matching_entries

def search_disease_keywords(dataset, keywords, sort_column='TARIF KELAS 1', ascending=False):
    all_matches = pd.DataFrame()
    for key in keywords:
        matches = dataset[dataset['DESKRIPSI KODE INA-CBG'].str.contains(key, case=False)]
        all_matches = pd.concat([all_matches, matches])
    return all_matches.sort_values(by=sort_column, ascending=ascending)

# PENGEMBANGAN
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def execute_query(query, params=None):
    connection = get_db_connection()
    if connection is None or not connection.is_connected():
        raise OperationalError("Database connection not available")

    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query, params)
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return result
    except OperationalError as e:
        print(f"OperationalError during query execution: {e}")
        raise
    except Error as e:
        print(f"Error during query execution: {e}")
        raise
    finally:
        if connection.is_connected():
            connection.close()

@app.route('/get_user_ids')
def get_user_ids():
    try:
        result = execute_query("SELECT DISTINCT user_id FROM query_disease")
        user_ids = [row["user_id"] for row in result]
        return jsonify(user_ids)
    except OperationalError as e:
        print(f"OperationalError: {e}")
        return jsonify({"error": "Operational error, please try again later"}), 500
    except Error as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 500

@app.route('/get_patient_names', methods=['GET'])
def get_patient_names():
    user_id = request.args.get('user_id')
    try:
        db_connection = mysql.connector.connect(
            host="34.145.29.172",
            user="beingman",
            password="123",
            database="oetomo"
        )
        cursor = db_connection.cursor()
        select_query = "SELECT DISTINCT patient_name FROM query_disease WHERE user_id = %s"
        cursor.execute(select_query, (user_id,))
        result = cursor.fetchall()
        patient_names = [row[0] for row in result]

        cursor.close()
        db_connection.close()

        return jsonify(patient_names)
    except mysql.connector.Error as err:
        return jsonify({'error': str(err)}), 500

@app.route('/get_queries')
def get_queries():
    user_id = request.args.get('user_id')
    patient_name = request.args.get('patient_name')

    if not user_id and not patient_name:
        return jsonify({"error": "User ID or Patient Name is required"}), 400

    try:
        query = "SELECT DISTINCT query FROM query_disease WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)

        if patient_name:
            query += " AND patient_name = %s"
            params.append(patient_name)

        result = execute_query(query, tuple(params))
        queries = [row["query"] for row in result]
        return jsonify(queries)
    except OperationalError as e:
        print(f"OperationalError: {e}")
        return jsonify({"error": "Operational error, please try again later"}), 500
    except SQLAlchemyError as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 500

@app.route('/get_icd_codes')
def get_icd_codes():
    user_id = request.args.get('user_id')
    patient_name = request.args.get('patient_name')
    query_param = request.args.get('query')

    if not user_id and not patient_name:
        return jsonify({"error": "User ID or Patient Name is required"}), 400

    try:
        query = "SELECT DISTINCT kode_icd FROM selected_icd WHERE 1=1"
        params = []

        if user_id:
            query += " AND user_id = %s"
            params.append(user_id)

        if patient_name:
            query += " AND patient_name = %s"
            params.append(patient_name)

        if query_param:
            query += " AND query = %s"
            params.append(query_param)

        result = execute_query(query, tuple(params))
        icd_codes = [row["kode_icd"] for row in result]
        return jsonify(icd_codes)
    except OperationalError as e:
        print(f"OperationalError: {e}")
        return jsonify({"error": "Operational error, please try again later"}), 500
    except SQLAlchemyError as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred"}), 500
#PENGEMBANGAN

#KLAIM INACBG
# Fungsi untuk menghubungkan ke database dan mengambil data dari tabel klaim_inacbg
def get_data_from_db():
    connection = mysql.connector.connect(
        host="34.145.29.172",
        user="beingman",
        password="123",
        database="oetomo"
    )
    query = "SELECT * FROM klaim_inacbg"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

# Fungsi pencarian
def search_df(df, search_term):
    result = df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
    return df[result]

@app.route('/search', methods=['GET'])
def search():
    search_term = request.args.get('q')
    if not search_term:
        return jsonify({"error": "Parameter 'q' is required"}), 400

    # Ambil data dari database
    df = get_data_from_db()

    # Lakukan pencarian
    result_df = search_df(df, search_term)

    # Konversi hasil ke JSON
    result_json = result_df.to_dict(orient='records')

    return jsonify(result_json)
#KLAIM INACBG

# insert_query_to_db(1, 'Test 1', 'John')
# insert_selected_icd_to_db(1, 'A00', 'Test 1', 'John')

if __name__ == '__main__':
    # app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
    socketio.run(app, port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
