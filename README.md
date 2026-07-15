# SMS Spam Detection - Multi-Model Classifier

Aplikasi web interaktif untuk deteksi spam SMS menggunakan 5 algoritma machine learning yang berbeda.

## Author
**NIM:** 230411100058  
**Mata Kuliah:** Pemrosesan Bahasa Alami (Natural Language Processing)  
**Project:** Tugas UAS - SMS Spam Detection  
**Tahun:** 2026

## Deskripsi Project

Aplikasi ini menggunakan Natural Language Processing (NLP) dan Machine Learning untuk mengklasifikasikan pesan SMS sebagai SPAM atau HAM (pesan legitimate). Sistem ini dilatih menggunakan dataset berisi 5,572 pesan SMS dengan akurasi hingga 98%.

## Fitur Utama

- **Real-time Prediction**: Prediksi instan untuk pesan individual
- **Multiple Models**: 5 algoritma ML berbeda untuk dipilih
- **Batch Processing**: Upload CSV untuk prediksi massal
- **Performance Metrics**: Visualisasi akurasi, precision, recall, F1-score
- **Data Analytics**: Word clouds, statistik, dan visualisasi data
- **Modern UI**: Interface web yang responsif dan user-friendly
- **Export Results**: Download hasil prediksi dalam format CSV

## Model Machine Learning

1. **Logistic Regression** - Baseline classifier yang cepat dan interpretable
2. **Support Vector Machine (SVM)** - Efektif untuk high-dimensional data
3. **Random Forest** - Ensemble method yang robust
4. **Gradient Boosting** - Sequential boosting untuk akurasi tinggi
5. **Multinomial Naive Bayes** - Probabilistic classifier untuk text data

## Dataset

- **Total Messages:** 5,572 SMS
- **Ham (Legitimate):** 4,825 messages (86.6%)
- **Spam:** 747 messages (13.4%)
- **Source:** SMS Spam Collection Dataset
- **Format:** CSV file dengan kolom label dan message

## Instalasi

### 1. Clone atau Download Project

```bash
cd sms-spam-detection-main
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data (Otomatis)

NLTK data akan didownload otomatis saat pertama kali menjalankan aplikasi.

## Cara Menjalankan

### Training Models (Opsional - jika belum ada model)

Jika folder `models/` kosong atau Anda ingin melatih ulang model:

```bash
python train_models.py
```

**Catatan:** Proses training memakan waktu 1-2 jam karena menggunakan GridSearchCV dengan 5-fold cross-validation.

### Menjalankan Aplikasi Web

```bash
streamlit run app_streamlit.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## Struktur Project

```
sms-spam-detection-main/
│
├── app_streamlit.py          # Aplikasi web utama (Streamlit)
├── train_models.py            # Script untuk training models
├── requirements.txt           # Dependencies Python
├── README.md                  # Dokumentasi ini
├── banner.jpg                 # Banner untuk UI
│
├── dataset/
│   └── spam.csv              # Dataset SMS (5,572 messages)
│
├── models/                    # Trained ML models (.joblib)
│   ├── logistic_regression_model.joblib
│   ├── svm_model.joblib
│   ├── random_forest_model.joblib
│   ├── gradient_boosting_model.joblib
│   └── multinomial_nb_model.joblib
│
└── reports/                   # Performance reports (.csv)
    ├── logistic_regression_report.csv
    ├── logistic_regression_confusion_matrix.csv
    ├── (dan file lainnya untuk setiap model)
    └── ...
```

## Penggunaan Aplikasi

### 1. Dashboard
- Melihat overview dataset dan statistik
- Distribusi spam vs ham
- Informasi model yang tersedia

### 2. Spam Prediction
- Input pesan SMS secara manual
- Pilih model yang ingin digunakan
- Lihat hasil prediksi dengan confidence score
- Indikator spam patterns yang terdeteksi

### 3. Model Performance
- Bandingkan performa 5 model
- Lihat classification report lengkap
- Confusion matrix untuk setiap model
- Visualisasi akurasi

### 4. Data Analytics
- Statistik panjang pesan
- Word clouds untuk spam dan ham
- Top keywords analysis
- Distribusi panjang pesan

### 5. Batch Prediction
- Upload file CSV dengan kolom 'message'
- Prediksi multiple messages sekaligus
- Download hasil prediksi
- Visualisasi hasil batch

### 6. About
- Informasi lengkap tentang project
- Penjelasan setiap model ML
- Pipeline preprocessing
- Referensi dan teknologi

## Format CSV untuk Batch Prediction

File CSV harus memiliki kolom `message`:

```csv
message
"Congratulations! You won a prize. Call now!"
"Hey, are you coming to the party tonight?"
"URGENT: Your account needs verification."
```

## Preprocessing Pipeline

1. **Lowercasing**: Konversi ke huruf kecil
2. **Remove Special Characters**: Hapus karakter non-alphanumeric
3. **Tokenization**: Pisahkan teks menjadi kata-kata
4. **Stopwords Removal**: Hapus kata umum (the, is, a, dll)
5. **Lemmatization**: Ubah kata ke bentuk dasar (running → run)
6. **TF-IDF Vectorization**: Konversi text ke numerical features

## Teknologi yang Digunakan

- **Python 3.x** - Programming language
- **Streamlit** - Web framework untuk UI
- **Scikit-learn** - Machine learning algorithms
- **NLTK** - Natural language processing
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **WordCloud** - Word cloud generation
- **Joblib** - Model serialization

## Performance

Model terbaik mencapai akurasi hingga **98%** pada test set dengan:
- **Precision**: 97-98% untuk deteksi spam
- **Recall**: 79-89% untuk deteksi spam
- **F1-Score**: 87-93% untuk deteksi spam

## Troubleshooting

### Error: Model files not found
- Jalankan `python train_models.py` untuk melatih model terlebih dahulu

### Error: Dataset not found
- Pastikan file `spam.csv` ada di folder `dataset/`

### Error: NLTK data not found
- Jalankan dalam Python:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

### Streamlit lemot atau tidak loading
- Refresh browser (Ctrl+R atau F5)
- Stop dan restart aplikasi
- Clear Streamlit cache: menu → Clear Cache

## Kontribusi

Project ini dibuat untuk tugas UAS mata kuliah Pemrosesan Bahasa Alami.

## Lisensi

Project ini dibuat untuk keperluan akademik.

## Pembuat

**Nama:** Wahyu Pratama
**NIM:** 230411100058  
**Project:** SMS Spam Detection  
**Tahun:** 2026

---

