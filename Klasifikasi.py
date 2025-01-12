import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from keras.models import load_model
import streamlit as st
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Menambahkan Font Awesome ke sidebar (untuk ikon)
# Menambahkan Font Awesome ke sidebar (untuk ikon)
st.sidebar.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">',
    unsafe_allow_html=True
)


# Membuat opsi menu dengan ikon
st.sidebar.markdown(
    """
    <h5 style="text-align: justify; color: black; margin-bottom: 10px;">
        KLASIFIKASI JAMU MADURA MENGGUNAKAN 
        METODE GATED RECURRENT UNIT (GRU) DENGAN TF-IDF UNTUK REPRESENTASI TEKS
    </h5>
    """,
    unsafe_allow_html=True
)

st.sidebar.image('menu.jpg', width=250)


selected = st.sidebar.radio(
    "MENU",
    [
        "🏢 Beranda",
        "📄 Data Jamu Madura",
        "🛠️ Data Hasil Preprocessing",
        "📊 Hasil TF-IDF",
        "📈 Akurasi",
        "📚 Hasil Prediksi Data (CSV)",
        "⚙️ Implementasi",
        "👤 Biodata Penulis",
        
    ]
)

# Content for each section
if selected == "🏢 Beranda":
    st.write("""
    <center><h3 style="text-align: justify;">
        KLASIFIKASI JAMU MADURA MENGGUNAKAN METODE 
        <span style="color:red;">GATED RECURRENT UNIT (GRU)</span> 
        DENGAN 
        <span style="color:red;">TF-IDF</span> 
        UNTUK REPRESENTASI TEKS
    </h3></center>
""", unsafe_allow_html=True)

     # Membuat tiga kolom, dengan kolom tengah lebih lebar
    col1, col2, col3 = st.columns([1, 1, 1])  # Kolom tengah lebih besar

    # Menampilkan gambar di kolom tengah
    with col1:   
        st.image('jamu1.png', width=200, caption=' Contoh Jamu Kesehatan')
    with col2:   
        st.image('jamu2.jpg', width=200, caption=' Contoh Jamu Perawatan Kewanitaan')
    with col3:
        st.image('jamu3.png', width=200, caption=' Contoh Jamu Pasutri')

 
    st.write("""
    <div style="text-align: justify;">
           Madura yang terletak di provinsi Jawa Timur, dikenal dengan kekayaan alam dan warisan budayanya yang unik, salah satunya adalah Jamu Madura. Jamu Madura  merupakan minuman tradisional yang terbuat dari bahan alami dan memiliki berbagai manfaat. Meskipun popularitas Jamu Madura tinggi, tantangan utama dalam proses klasifikasi jamu  adalah variabilitas khasiatnya dan ketergantungan pada pengetahuan tradisional, sering kali tidak terdokumentasi secara sistematis. Penelitian ini bertujuan untuk mengatasi tantangan tersebut dengan mengembangkan metode otomatis untuk klasifikasi Jamu Madura berdasarkan atribut khasiat jamu madura menggunakan teknik Gated Recurrent Units (GRU) yang didukung oleh representasi teks TF-IDF. Dimana pada penelitian ini atribut khasiat merupakan atribut yang digunakan.  Dalam penelitian ini, Peneliti mengimplementasikan Jamu Madura pada tiga kategori (1) Jamu Kesehatan, (2) Jamu Perawatan Kewanitaan,  (3) Jamu Pasutri. Ketiga kategori akan di validasi oleh pihak yang ahli dibidangnya. TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengubah informasi tekstual menjadi numerik. GRU adalah jenis  jaringan saraf berulang  digunakan dalam pemrosesan dan klasifikasi teks. GRU dirancang mengatasi beberapa keterbatasan dari RNN, selanjutnya TF-IDF   adalah teknik yang umum digunakan pada pemrosesan bahasa alami (NLP) untuk menilai kepentingan / bobot suatu kata dalam dokumen relatif terhadap koleksi dokumen. Penelitian ini diharapkan dapat meningkatkan akurasi dalam klasifikasi  jamu madura menggunakan metode GRU dengan Representasi Teks TF-IDF.
    </div>
    """, unsafe_allow_html=True)

    
elif selected == "📄 Data Jamu Madura":
    st.subheader("Data Jamu Madura")
    st.write("""
    <div style="text-align: justify;">
           Data Jamu Madura terdiri dari atribut khasiat yang digunakan sebagai acuan dalam proses klasifikasi Jamu Madura menggunakan metode GRU dengan representasi teks berbasis TF-IDF. Kategori klasifikasi Jamu Madura terbagi menjadi tiga, yaitu: 1) Jamu Kesehatan, 2) Jamu Perawatan Kewanitaan, dan 3) Jamu Pasutri. Pelabelan kategori ini diperoleh dari penelitian sebelumnya dan telah divalidasi oleh para ahli di bidangnya.
    </div>
    """, unsafe_allow_html=True)

    data_jamu = pd.read_csv('https://raw.githubusercontent.com/JhonFiliusGultom/Skripsi/refs/heads/main/Data_Skripsi_J-Madura_.csv')
    columns_to_drop = ['Produsen']
    data_jamu = data_jamu.drop(columns=columns_to_drop, errors='ignore')
    st.dataframe(data_jamu)

    Kategori_count = data_jamu['Kategori'].value_counts()

# Menampilkan jumlah data per kategori di Streamlit

# Visualisasi menggunakan seaborn (Bar Plot)
    plt.figure(figsize=(8,6))
    sns.barplot(x=Kategori_count.index, y=Kategori_count.values, palette='viridis')
    plt.title('Jumlah Data per Kategori')
    plt.xlabel('Kategori')
    plt.ylabel('Jumlah')
    plt.xticks(rotation=45)  # Untuk rotasi label sumbu x jika diperlukan

# Menampilkan plot di Streamlit
    st.pyplot(plt)

elif selected == "🛠️ Data Hasil Preprocessing":
    st.subheader("Preprocessing")
    
    st.write("""
    <div style="text-align: justify;">
            Pra proses (Preprocessing) data merupakan  suatu teknik algoritma penting dalam melakukan suatu proses analisis klasifikasi teks. Dimana preprocessing memiliki tujuan untuk melakukan pembersihan suatu data dari unsur-unsur yang tidak dibutuhkan untuk mempercepat proses klasifikasi. 
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Tahapan Preprocessing")
    
    st.write("""
    <div style="text-align: justify;">
            1. <b> Case Folding </b> : Tahapan pertama yaitu case folding dimana tahap ini semua huruf diubah menjadi huruf kecil atau huruf besar dan angka, dan simbol khusus. 
    </div>
    """, unsafe_allow_html=True)
    st.write("""
    <div style="text-align: justify;">
            2. <b> Puncation Removal </b>:  merupakan suatu tahapan proses dimana melakukan pembersihan data dari karakter dan elemen yang tidak relevan yang mengganggu analisis, seperti tanda baca.
    </div>
    """, unsafe_allow_html=True)
    st.write("""
    <div style="text-align: justify;">
            3. <b> Tokenizing </b> : Tahapan ini merupakan tahapan selanjutnya yaitu tokenizing dimana merupakan suatu proses memisahkan / memecahkan suatu teks menjadi unit-unit yang lebih kecil yang disebut token. 
    </div>
    """, unsafe_allow_html=True)
    st.write("""
    <div style="text-align: justify;">
            4. <b> Stopword Removal </b> :  merupakan proses menghilangkan kata-kata yang sering muncul dalam dokumen tetapi tidak memiliki arti penting dan tidak memiliki pengaruh.
    </div>
    """, unsafe_allow_html=True)
    st.write("""
    <div style="text-align: justify;">
            5. <b> Stemming </b> : digunakan untuk menemukan akar atau inti  dari setiap kata dan menghapus imbuhan seperti awalan, sisipan, dan akhiran.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Data Hasil Preprocessing")
    preprocessing = pd.read_csv('https://raw.githubusercontent.com/JhonFiliusGultom/Skripsi/refs/heads/main/datajamu_preprocessingg.csv')
    columns_to_drop = ['Kategori', 'Produsen', 'Nama']  # Nama kolom yang akan dihapus
    preprocessing = preprocessing.drop(columns=columns_to_drop, errors='ignore')
    st.dataframe(preprocessing)

elif selected == "📊 Hasil TF-IDF":
    st.subheader("TF-IDF (Term Frequency Inverse Document Frequency)")
    
    st.write("""
    <div style="text-align: justify;">
           Pada penelitian ini, metode pembobotan, Term Frequency (TF) dan Inverse Document Frequency (IDF) , Term Frequency – Inverse Document Frequency (TF-IDF) adalah suatu algoritma yang digunakan untuk menampilkan frekuensi kata yang keluar dalam dokumen. Term Frequency (TF) adalah frekuensi munculnya kata dalam suatu dokumen, sedangkan Document Frequency (DF) adalah banyaknya dokumen yang mengandung kata tertentu. Nilai bobot didapat dari TF dan Inverse dari DF. 
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Hasil TF-IDF")
    tf_idf = pd.read_csv('https://raw.githubusercontent.com/JhonFiliusGultom/Skripsi/refs/heads/main/tfidf__.csv')
    st.dataframe(tf_idf)

elif selected == "📈 Akurasi":
    # Skenario 1
    data = {'Parameter': ['Epoch 10, Batch Size 32, Learning Rate 0,0009', 
                          'Epoch 15, Batch Size 32, Learning Rate 0,0009', 
                          'Epoch 20, Batch Size 32, Learning Rate 0,0009', 
                          'Epoch 25, Batch Size 32, Learning Rate 0,0009',  # Koma ditambahkan di sini
                          'Epoch 30, Batch Size 32, Learning Rate 0,0009'],
            'Akurasi': [92.73, 89.09, 90.91, 90.91, 90.91]}
    df_akurasi = pd.DataFrame(data)

    # Menampilkan judul grafik untuk Skenario 1
    st.markdown("<h6 style='color:black;'>Grafik Akurasi Tertinggi pada masing-masing Skenario pada Klasifikasi Jamu Madura Menggunakan Metode <span style='color:red;'>Gated Recurrent Unit (GRU)<span style='color:black;'> Dengan<span style='color:red;'> TF-IDF <span style='color:black;'> Untuk Representasi Teks</span></h1>", unsafe_allow_html=True)
    
    # Menampilkan grafik untuk Skenario 1
    st.bar_chart(df_akurasi.set_index('Parameter'))

elif selected == "📚 Hasil Prediksi Data (CSV)":
        # Fungsi preprocessing
    def casefoldingkhasiat(data_jamu):
        return data_jamu.lower()

    def remove_punctuation(khasiat):
        return khasiat.translate(str.maketrans("", "", string.punctuation))

    def tokenize_text(khasiat):
        tokens = nltk.word_tokenize(khasiat)
        return tokens

    # Load stopwords dan stemmer
    nltk.download('stopwords')
    nltk.download('punkt')
    list_stopwords = nltk.corpus.stopwords.words('indonesian')
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Fungsi TF, IDF, dan TF-IDF
    def compute_tf(doc):
        words = doc.split()
        total_terms = len(words)
        tf_count = {}
        for word in words:
            tf_count[word] = tf_count.get(word, 0) + 1
        return {term: count / total_terms for term, count in tf_count.items()}

    def compute_tfidf(tf, idf):
        return {term: tf.get(term, 0) * idf[term] for term in idf}

    # Load model dan IDF
    model = load_model("jamu_gru_model.h5")
    idf = joblib.load("idf.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Fungsi untuk melakukan prediksi
    def predict_category(new_text):
        # Preprocessing teks baru
        new_case_folding = casefoldingkhasiat(new_text)
        new_no_punctuation = remove_punctuation(new_case_folding)
        new_tokens = tokenize_text(new_no_punctuation)
        new_no_stopwords = [word for word in new_tokens if word not in list_stopwords]
        new_stemmed = [stemmer.stem(word) for word in new_no_stopwords]
        new_stemmed_text = ' '.join(new_stemmed)

        # Hitung TF untuk teks baru
        new_tf = compute_tf(new_stemmed_text)

        # Hitung TF-IDF untuk teks baru
        new_tfidf = compute_tfidf(new_tf, idf)

        # Ubah menjadi DataFrame
        new_tfidf_df = pd.DataFrame([new_tfidf]).fillna(0)

        # Ubah ke array dan tambahkan dimensi timesteps
        new_tfidf_array = np.expand_dims(new_tfidf_df.values, axis=1)

        # Prediksi menggunakan model
        predictions = model.predict(new_tfidf_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Kembalikan kategori yang diprediksi
        return label_encoder.inverse_transform(predicted_class)
    
    def process_and_predict_csv(uploaded_file):
        # Membaca file CSV
        data = pd.read_csv(uploaded_file)
        
        if 'Khasiat' not in data.columns:
            st.error("File CSV harus memiliki kolom 'Deskripsi Khasiat'")
            return None

        predictions = []
        
        for khasiat in data['Khasiat']:
            try:
                prediction = predict_category(khasiat)[0]
                predictions.append(prediction)
            except Exception as e:
                predictions.append(f"Error: {e}")
        
        data['Prediksi Kategori'] = predictions
        return data

    st.markdown("<h3>Hasil Prediksi Data Jamu Madura</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Unggah file CSV dengan kolom 'Deskripsi Khasiat'", type="csv")

    if uploaded_file:
        result_data = process_and_predict_csv(uploaded_file)
        if result_data is not None:
            st.success("Prediksi selesai! Berikut adalah hasilnya:")
            st.dataframe(result_data)

            csv_result = result_data.to_csv(index=False)
            st.download_button(
                label="Unduh Hasil Prediksi",
                data=csv_result,
                file_name="hasil_prediksi_jamu.csv",
                mime="text/csv"
            )

elif selected == "⚙️ Implementasi":
    # Fungsi preprocessing
    def casefoldingkhasiat(data_jamu):
        return data_jamu.lower()

    def remove_punctuation(khasiat):
        return khasiat.translate(str.maketrans("", "", string.punctuation))

    def tokenize_text(khasiat):
        tokens = nltk.word_tokenize(khasiat)
        return tokens

    # Load stopwords dan stemmer
    nltk.download('stopwords')
    nltk.download('punkt')
    list_stopwords = nltk.corpus.stopwords.words('indonesian')
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Fungsi TF, IDF, dan TF-IDF
    def compute_tf(doc):
        words = doc.split()
        total_terms = len(words)
        tf_count = {}
        for word in words:
            tf_count[word] = tf_count.get(word, 0) + 1
        return {term: count / total_terms for term, count in tf_count.items()}

    def compute_tfidf(tf, idf):
        return {term: tf.get(term, 0) * idf[term] for term in idf}

    # Load model dan IDF
    model = load_model("jamu_gru_model.h5")
    idf = joblib.load("idf.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Fungsi untuk melakukan prediksi
    def predict_category(new_text):
        # Preprocessing teks baru
        new_case_folding = casefoldingkhasiat(new_text)
        new_no_punctuation = remove_punctuation(new_case_folding)
        new_tokens = tokenize_text(new_no_punctuation)
        new_no_stopwords = [word for word in new_tokens if word not in list_stopwords]
        new_stemmed = [stemmer.stem(word) for word in new_no_stopwords]
        new_stemmed_text = ' '.join(new_stemmed)

        # Hitung TF untuk teks baru
        new_tf = compute_tf(new_stemmed_text)

        # Hitung TF-IDF untuk teks baru
        new_tfidf = compute_tfidf(new_tf, idf)

        # Ubah menjadi DataFrame
        new_tfidf_df = pd.DataFrame([new_tfidf]).fillna(0)

        # Ubah ke array dan tambahkan dimensi timesteps
        new_tfidf_array = np.expand_dims(new_tfidf_df.values, axis=1)

        # Prediksi menggunakan model
        predictions = model.predict(new_tfidf_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Kembalikan kategori yang diprediksi
        return label_encoder.inverse_transform(predicted_class)

    # Streamlit interface untuk implementasi
    st.markdown("<h1 style='color:black;'>Implementasi Klasifikasi <span style='color:red;'>Jamu Madura</span></h1>", unsafe_allow_html=True)
    st.write("Masukkan deskripsi khasiat jamu, dan sistem akan mengklasifikasikan kategori jamu tersebut.")

    # Input dari pengguna
    user_input = st.text_area("Masukkan Deskripsi Khasiat Jamu:", "")

    if st.button("Klasifikasikan"):
        if user_input.strip():
            prediction = predict_category(user_input)  # Fungsi untuk memprediksi kategori
            if prediction[0] == "Jamu Kesehatan":
                st.success(f"Hasil Klasifikasi: {prediction[0]}")
            elif prediction[0] == "Jamu Perawatan Kewanitaan":
                st.info(f"Hasil Klasifikasi: {prediction[0]}")
            elif prediction[0] == "Jamu Pasutri":
                st.error(f"Hasil Klasifikasi: {prediction[0]}")
        else:
            st.warning("Harap masukkan deskripsi khasiat jamu sebelum mengklik tombol klasifikasikan!")

## Biodata Penulis
if selected == "👤 Biodata Penulis":
    
    # Menampilkan Daftar Riwayat Hidup
    st.write("""
    <center><h3 style="font-family: 'Arial', sans-serif; color: #2E3B4E; font-weight: bold;">DAFTAR RIWAYAT HIDUP</h3></center>
""", unsafe_allow_html=True)

    st.write("""
    **Data Pribadi:**
    - **Nama:** Jhon Filius Gultom
    - **Jenis Kelamin:** Laki-Laki
    - **Alamat:** Jl. Rakutta Sembiring GG. Kenali, Pematangsiantar, Sumatera Utara
    - **Agama:** Katolik
    - **Kewarganegaraan:** Indonesia
    - **Program Studi:** Teknik Informatika
    - **Email:** johnfliusgultom@gmail.com
    """)

    # Menampilkan Riwayat Pendidikan dalam bentuk tabel
    st.write("""
    **Riwayat Pendidikan:**
    """)
    st.write("""
    <table style="width: 100%; border-collapse: collapse; text-align: left; padding: 15px;">
        <tr style="background-color: #f4f4f4;">
            <th style="padding: 10px; border: 1px solid #ddd;">No</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Pendidikan Formal</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Tahun</th>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">1</td>
            <td style="padding: 10px; border: 1px solid #ddd;">SDN 122374 Pematangsiantar</td>
            <td style="padding: 10px; border: 1px solid #ddd;">2009-2015</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">2</td>
            <td style="padding: 10px; border: 1px solid #ddd;">SMPN 8 Pematangsiantar</td>
            <td style="padding: 10px; border: 1px solid #ddd;">2015-2018</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">3</td>
            <td style="padding: 10px; border: 1px solid #ddd;">SMK Swasta Assisi Siantar</td>
            <td style="padding: 10px; border: 1px solid #ddd;">2018-2021</td>
        </tr>
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;">4</td>
            <td style="padding: 10px; border: 1px solid #ddd;">Universitas Trunojoyo Madura</td>
            <td style="padding: 10px; border: 1px solid #ddd;">2021-2024</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.pyplot(plt)