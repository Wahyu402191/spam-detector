"""
Smart SMS Spam Detection Application
Author: 230411100058
Description: Complete web application for SMS spam detection with multiple ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import re
import os
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# ==================== PREPROCESSING FUNCTIONS ====================

def preprocess_text(text):
    """Preprocess SMS text for model prediction"""
    if not text or not isinstance(text, str):
        return ""
    
    lemmatizer = WordNetLemmatizer()
    cleaned = re.sub('[^a-zA-Z0-9]', ' ', text)
    cleaned = cleaned.lower()
    words = cleaned.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def get_spam_indicators(text):
    """Identify spam indicators in text"""
    indicators = []
    text_lower = text.lower()
    
    spam_keywords = ['free', 'win', 'winner', 'cash', 'prize', 'claim', 'urgent', 
                     'congratulations', 'selected', 'reward', 'call now']
    
    for keyword in spam_keywords:
        if keyword in text_lower:
            indicators.append(keyword)
    
    if text.count('!') > 2:
        indicators.append('excessive_exclamation')
    if re.search(r'\d{10,}', text):
        indicators.append('phone_number')
    if re.search(r'http[s]?://|www\.', text):
        indicators.append('url_link')
    
    return indicators

# ==================== PAGE CONFIGURATION ====================

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================

st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .stat-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }

    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #666;
        margin: 0.5rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result boxes */
    .result-spam {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
    }
    
    .result-ham {
        background: linear-gradient(135deg, #51cf66, #37b24d);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(81, 207, 102, 0.3);
    }

    
    /* Confidence bar */
    .confidence-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .confidence-bar {
        height: 30px;
        background: #e9ecef;
        border-radius: 15px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        transition: width 0.5s ease;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f5ff;
        border-left: 4px solid #339af0;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    
    .warning-box {
        background: #fff3bf;
        border-left: 4px solid #ffd43b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d3f9d8;
        border-left: 4px solid #51cf66;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Text area styling */
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Titles */
    h1 {
        color: #2d3436;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #2d3436;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==================== LOAD MODELS AND DATA ====================

@st.cache_resource
def load_all_models():
    """Load all trained ML models"""
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression_model.joblib',
        'SVM': 'models/svm_model.joblib',
        'Random Forest': 'models/random_forest_model.joblib',
        'Multinomial NB': 'models/multinomial_nb_model.joblib'
    }
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                models[name] = load(filepath)
            except Exception as e:
                st.warning(f"Model {name} tidak dapat dimuat (version mismatch)")
        else:
            st.warning(f"File model tidak ditemukan: {filepath}")
    
    return models

@st.cache_data
def load_dataset_statistics():
    """Load and calculate dataset statistics"""
    try:
        df = pd.read_csv('dataset/spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        
        stats = {
            'total': len(df),
            'spam_count': len(df[df['label'] == 'spam']),
            'ham_count': len(df[df['label'] == 'ham']),
            'spam_percentage': (len(df[df['label'] == 'spam']) / len(df)) * 100,
            'ham_percentage': (len(df[df['label'] == 'ham']) / len(df)) * 100
        }
        return stats, df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None, None


@st.cache_data
def load_model_reports():
    """Load classification reports for all models"""
    reports = {}
    confusion_matrices = {}
    
    report_files = {
        'Logistic Regression': 'reports/logistic_regression',
        'SVM': 'reports/svm',
        'Random Forest': 'reports/random_forest',
        'Multinomial NB': 'reports/multinomial_nb'
    }
    
    for name, filepath in report_files.items():
        report_path = f"{filepath}_report.csv"
        cm_path = f"{filepath}_confusion_matrix.csv"
        
        if os.path.exists(report_path):
            reports[name] = pd.read_csv(report_path, index_col=0)
        
        if os.path.exists(cm_path):
            confusion_matrices[name] = pd.read_csv(cm_path, index_col=0)
    
    return reports, confusion_matrices

# Load everything
models = load_all_models()
dataset_stats, dataset_df = load_dataset_statistics()
reports, confusion_matrices = load_model_reports()

# ==================== SIDEBAR NAVIGATION ====================

st.sidebar.title("SMS Spam Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi:",
    ["Dashboard", "Spam Prediction", "Model Performance", 
     "Data Analytics", "Batch Prediction", "About"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Multi-Model Classifier**

Menggunakan 4 algoritma ML:
- Logistic Regression
- SVM
- Random Forest
- Naive Bayes

**Mahasiswa:**  
Wahyu Pratama  
NIM: 230411100058
""")



# ==================== PAGE 1: DASHBOARD ====================

if page == "Dashboard":
    st.title("SMS Spam Detection Dashboard")
    st.markdown("---")
    
    st.markdown("""
    ### Selamat Datang di Smart SMS Spam Detector
    
    Aplikasi ini menggunakan **4 model machine learning** yang berbeda untuk mengklasifikasikan 
    pesan SMS sebagai **SPAM** atau **HAM** (pesan legitimate). Sistem telah dilatih menggunakan 
    ribuan pesan SMS nyata untuk memberikan prediksi yang akurat.
    """)
    
    if dataset_stats:
        st.markdown("### Ringkasan Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number">{dataset_stats['total']:,}</p>
                <p class="stat-label">Total Messages</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number" style="color: #51cf66;">{dataset_stats['ham_count']:,}</p>
                <p class="stat-label">Ham (Legitimate)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number" style="color: #ff6b6b;">{dataset_stats['spam_count']:,}</p>
                <p class="stat-label">Spam Messages</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <p class="stat-number">5</p>
                <p class="stat-label">ML Models</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        
        # Distribution charts
        st.markdown("### Distribusi Pesan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#51cf66', '#ff6b6b']
            explode = (0.05, 0.05)
            
            ax.pie(
                [dataset_stats['ham_count'], dataset_stats['spam_count']],
                labels=['Ham', 'Spam'],
                colors=colors,
                autopct='%1.1f%%',
                explode=explode,
                shadow=True,
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            ax.set_title('Ham vs Spam Distribution', fontsize=14, fontweight='bold', pad=20)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            categories = ['Ham', 'Spam']
            counts = [dataset_stats['ham_count'], dataset_stats['spam_count']]
            
            bars = ax.bar(categories, counts, color=colors, width=0.6, edgecolor='white', linewidth=2)
            ax.set_ylabel('Number of Messages', fontsize=12, fontweight='bold')
            ax.set_title('Message Count Comparison', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, max(counts) * 1.15)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
    
    # Model Information
    st.markdown("### Model yang Tersedia")
    
    models_info = {
        'Logistic Regression': 'Classifier linear yang cepat dan mudah diinterpretasi',
        'SVM': 'Support Vector Machine untuk decision boundary yang kompleks',
        'Random Forest': 'Ensemble dari decision trees untuk prediksi yang robust',
        'Multinomial NB': 'Classifier probabilistik berdasarkan Bayes theorem'
    }

    
    col1, col2 = st.columns(2)
    
    for i, (model_name, description) in enumerate(models_info.items()):
        with col1 if i % 2 == 0 else col2:
            with st.container():
                st.markdown(f"""
                <div class="info-box">
                    <strong>{model_name}</strong><br>
                    {description}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("### Panduan Cepat")
    st.markdown("""
    1. Buka halaman **Spam Prediction** untuk menguji pesan individual
    2. Lihat **Model Performance** untuk membandingkan metrik akurasi
    3. Jelajahi **Data Analytics** untuk insights dan visualisasi
    4. Gunakan **Batch Prediction** untuk mengklasifikasikan banyak pesan sekaligus
    """)

# ==================== PAGE 2: SPAM PREDICTION ====================

elif page == "Spam Prediction":
    st.title("Spam Prediction")
    st.markdown("---")
    
    st.markdown("""
    Masukkan pesan SMS di bawah ini untuk mengecek apakah pesan tersebut spam atau legitimate.
    Anda dapat memilih model yang berbeda untuk melihat bagaimana setiap model mengklasifikasikan pesan.
    
    **Perbedaan Model:**
    - **Logistic Regression**: Model dasar yang cepat, cocok untuk klasifikasi linear
    - **SVM**: Efektif untuk data berdimensi tinggi, mencari hyperplane optimal
    - **Random Forest**: Menggunakan banyak decision tree, lebih robust terhadap noise
    - **Naive Bayes**: Berbasis probabilitas, cepat dan efisien untuk text classification
    """)
    
    # Initialize session state for user input
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Masukkan Pesan SMS:",
            value=st.session_state.user_input,
            height=150,
            placeholder="Ketik atau paste pesan SMS Anda di sini..."
        )
    
    with col2:
        st.markdown("### Pengaturan")
        selected_model = st.selectbox(
            "Pilih Model:",
            list(models.keys())
        )
        
        st.markdown("### Contoh Cepat")
        if st.button("Coba Contoh Spam", use_container_width=True):
            st.session_state.user_input = "URGENT! You have won a $1000 cash prize! Call now to claim: 1-800-WIN-CASH"
            st.rerun()
        
        if st.button("Coba Contoh Ham", use_container_width=True):
            st.session_state.user_input = "Hey, are you free for dinner tonight? Let me know!"
            st.rerun()

    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        predict_button = st.button("Prediksi", use_container_width=True)
    
    with col2:
        if st.button("Hapus", use_container_width=True):
            st.session_state.user_input = ""
            st.rerun()
    
    if predict_button and user_input:
        with st.spinner("Menganalisis pesan..."):
            # Preprocess
            processed_text = preprocess_text(user_input)
            
            # Predict
            model = models[selected_model]
            prediction = model.predict([processed_text])[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([processed_text])[0]
                confidence = max(proba) * 100
            else:
                confidence = 100.0
            
            # Display result
            st.markdown("---")
            st.markdown("### Hasil Prediksi")
            
            if prediction == 'spam':
                st.markdown(f"""
                <div class="result-spam">
                    Pesan ini adalah SPAM
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ham">
                    Pesan ini adalah HAM (Legitimate)
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence score
            st.markdown("### Tingkat Kepercayaan")
            st.markdown(f"""
            <div class="confidence-container">
                <p style="margin: 0 0 0.5rem 0; font-weight: 600;">
                    Confidence Score: {confidence:.2f}%
                </p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence}%;">
                        {confidence:.1f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            
            # Spam indicators
            indicators = get_spam_indicators(user_input)
            
            if indicators:
                st.markdown("### Indikator Spam Terdeteksi")
                st.markdown("""
                <div class="warning-box">
                    <strong>Pola mencurigakan ditemukan:</strong><br>
                    {} 
                </div>
                """.format(", ".join(f"<code>{ind}</code>" for ind in indicators)), 
                unsafe_allow_html=True)
            
            # Processing details
            with st.expander("Detail Pemrosesan"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pesan Asli:**")
                    st.text(user_input[:200] + "..." if len(user_input) > 200 else user_input)
                    
                    st.markdown(f"**Jumlah Karakter:** {len(user_input)}")
                    st.markdown(f"**Jumlah Kata:** {len(user_input.split())}")
                
                with col2:
                    st.markdown("**Pesan Setelah Diproses:**")
                    st.text(processed_text[:200] + "..." if len(processed_text) > 200 else processed_text)
                    
                    st.markdown(f"**Model yang Digunakan:** {selected_model}")
                    st.markdown(f"**Hasil Prediksi:** {prediction.upper()}")
    
    elif predict_button and not user_input:
        st.warning("Silakan masukkan pesan SMS untuk dianalisis.")

# ==================== PAGE 3: MODEL PERFORMANCE ====================

elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.markdown("---")
    
    if reports:
        st.markdown("### Laporan Klasifikasi")
        
        # Comparison table
        comparison_data = []
        
        for model_name, report in reports.items():
            if 'accuracy' in report.index:
                accuracy = report.loc['accuracy', 'precision']
            else:
                accuracy = 0.0
            
            spam_precision = report.loc['spam', 'precision'] if 'spam' in report.index else 0.0
            spam_recall = report.loc['spam', 'recall'] if 'spam' in report.index else 0.0
            spam_f1 = report.loc['spam', 'f1-score'] if 'spam' in report.index else 0.0
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{accuracy:.4f}",
                'Spam Precision': f"{spam_precision:.4f}",
                'Spam Recall': f"{spam_recall:.4f}",
                'Spam F1-Score': f"{spam_f1:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        
        # Visual comparison
        st.markdown("### Perbandingan Akurasi")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = [item['Model'] for item in comparison_data]
        accuracies = [float(item['Accuracy']) for item in comparison_data]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
        bars = ax.barh(model_names, accuracies, color=colors, edgecolor='white', linewidth=2)
        
        ax.set_xlabel('Akurasi', fontsize=12, fontweight='bold')
        ax.set_title('Perbandingan Akurasi Model', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{acc:.4f}',
                   va='center', fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
        plt.close()
        
        # Detailed report for selected model
        st.markdown("---")
        st.markdown("### Laporan Performa Detail")
        
        selected_model_perf = st.selectbox(
            "Pilih Model untuk Laporan Detail:",
            list(reports.keys())
        )
        
        if selected_model_perf in reports:
            report_df = reports[selected_model_perf]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Metrik Klasifikasi")
                display_df = report_df.drop(columns=['support'], errors='ignore')
                st.dataframe(display_df, use_container_width=True)
            
            with col2:
                if selected_model_perf in confusion_matrices:
                    st.markdown("#### Confusion Matrix")
                    
                    cm_df = confusion_matrices[selected_model_perf]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm_df, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        cbar_kws={'label': 'Jumlah'},
                        linewidths=2,
                        linecolor='white',
                        ax=ax
                    )
                    ax.set_title(f'Confusion Matrix - {selected_model_perf}', 
                               fontsize=12, fontweight='bold', pad=10)
                    ax.set_xlabel('Label Prediksi', fontsize=10, fontweight='bold')
                    ax.set_ylabel('Label Aktual', fontsize=10, fontweight='bold')
                    
                    st.pyplot(fig)
                    plt.close()
    
    else:
        st.warning("Laporan performa model tidak ditemukan. Silakan latih model terlebih dahulu.")


# ==================== PAGE 4: DATA ANALYTICS ====================

elif page == "Data Analytics":
    st.title("Data Analytics & Insights")
    st.markdown("---")
    
    if dataset_df is not None:
        st.markdown("### Statistik Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_ham_length = dataset_df[dataset_df['label'] == 'ham']['message'].str.len().mean()
            st.metric("Rata-rata Panjang Ham", f"{avg_ham_length:.0f} karakter")
        
        with col2:
            avg_spam_length = dataset_df[dataset_df['label'] == 'spam']['message'].str.len().mean()
            st.metric("Rata-rata Panjang Spam", f"{avg_spam_length:.0f} karakter")
        
        with col3:
            length_diff = avg_spam_length - avg_ham_length
            st.metric("Selisih Panjang", f"{length_diff:.0f} karakter")
        
        # Message length distribution
        st.markdown("### Distribusi Panjang Pesan")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ham_lengths = dataset_df[dataset_df['label'] == 'ham']['message'].str.len()
        spam_lengths = dataset_df[dataset_df['label'] == 'spam']['message'].str.len()
        
        ax.hist(ham_lengths, bins=50, alpha=0.6, label='Ham', color='#51cf66', edgecolor='white')
        ax.hist(spam_lengths, bins=50, alpha=0.6, label='Spam', color='#ff6b6b', edgecolor='white')
        
        ax.set_xlabel('Panjang Pesan (karakter)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frekuensi', fontsize=12, fontweight='bold')
        ax.set_title('Distribusi Panjang Pesan', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        
        st.pyplot(fig)
        plt.close()
        
        # Word clouds
        st.markdown("### Word Clouds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Pesan Ham")
            ham_text = ' '.join(dataset_df[dataset_df['label'] == 'ham']['message'].astype(str))
            
            try:
                wordcloud_ham = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='Greens',
                    max_words=100
                ).generate(ham_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_ham, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            except:
                st.info("Tidak dapat membuat word cloud untuk pesan ham")

        
        with col2:
            st.markdown("#### Pesan Spam")
            spam_text = ' '.join(dataset_df[dataset_df['label'] == 'spam']['message'].astype(str))
            
            try:
                wordcloud_spam = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='Reds',
                    max_words=100
                ).generate(spam_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud_spam, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            except:
                st.info("Tidak dapat membuat word cloud untuk pesan spam")
        
        # Top keywords
        st.markdown("### Kata Kunci Teratas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Kata Paling Umum di Ham")
            ham_words = ' '.join(dataset_df[dataset_df['label'] == 'ham']['message'].astype(str)).lower().split()
            from collections import Counter
            ham_counter = Counter(ham_words)
            top_ham = pd.DataFrame(ham_counter.most_common(15), columns=['Word', 'Frequency'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(top_ham['Word'][::-1], top_ham['Frequency'][::-1], color='#51cf66', edgecolor='white', linewidth=2)
            ax.set_xlabel('Frekuensi', fontsize=11, fontweight='bold')
            ax.set_title('15 Kata Teratas di Pesan Ham', fontsize=12, fontweight='bold', pad=15)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Kata Paling Umum di Spam")
            spam_words = ' '.join(dataset_df[dataset_df['label'] == 'spam']['message'].astype(str)).lower().split()
            spam_counter = Counter(spam_words)
            top_spam = pd.DataFrame(spam_counter.most_common(15), columns=['Word', 'Frequency'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(top_spam['Word'][::-1], top_spam['Frequency'][::-1], color='#ff6b6b', edgecolor='white', linewidth=2)
            ax.set_xlabel('Frekuensi', fontsize=11, fontweight='bold')
            ax.set_title('15 Kata Teratas di Pesan Spam', fontsize=12, fontweight='bold', pad=15)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    else:
        st.warning("Dataset tidak ditemukan. Pastikan spam.csv ada di folder dataset.")


# ==================== PAGE 5: BATCH PREDICTION ====================

elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.markdown("---")
    
    st.markdown("""
    Upload file CSV yang berisi pesan SMS untuk mengklasifikasikan banyak pesan sekaligus.
    File CSV Anda harus memiliki kolom bernama **'message'** yang berisi teks SMS.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload file CSV",
            type=['csv'],
            help="Upload file CSV dengan kolom 'message'"
        )
    
    with col2:
        st.markdown("### Pengaturan")
        batch_model = st.selectbox(
            "Pilih Model:",
            list(models.keys())
        )
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, encoding='utf-8')
            
            st.markdown("### Preview Data yang Diupload")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            if 'message' not in df_upload.columns:
                st.error("File CSV harus memiliki kolom 'message'!")
            else:
                if st.button("Prediksi Semua Pesan", use_container_width=True):
                    with st.spinner("Memproses pesan..."):
                        # Preprocess all messages
                        df_upload['processed'] = df_upload['message'].apply(preprocess_text)
                        
                        # Predict
                        model = models[batch_model]
                        predictions = model.predict(df_upload['processed'])
                        
                        # Get probabilities if available
                        if hasattr(model, 'predict_proba'):
                            probas = model.predict_proba(df_upload['processed'])
                            confidences = [max(p) * 100 for p in probas]
                        else:
                            confidences = [100.0] * len(predictions)
                        
                        # Add results to dataframe
                        df_upload['prediction'] = predictions
                        df_upload['confidence'] = [f"{c:.2f}%" for c in confidences]
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### Hasil Prediksi")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        spam_count = sum(predictions == 'spam')
                        ham_count = sum(predictions == 'ham')
                        
                        with col1:
                            st.metric("Total Diproses", len(predictions))
                        with col2:
                            st.metric("Spam Terdeteksi", spam_count)
                        with col3:
                            st.metric("Ham Terdeteksi", ham_count)
                        
                        # Results table
                        st.markdown("### Hasil Detail")
                        results_df = df_upload[['message', 'prediction', 'confidence']]
                        st.dataframe(results_df, use_container_width=True, hide_index=True)

                        
                        # Download button
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Hasil CSV",
                            data=csv,
                            file_name="spam_prediction_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Visualization
                        st.markdown("### Visualisasi Hasil")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        categories = ['Ham', 'Spam']
                        counts = [ham_count, spam_count]
                        colors = ['#51cf66', '#ff6b6b']
                        
                        bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor='white', linewidth=2)
                        ax.set_ylabel('Jumlah', fontsize=12, fontweight='bold')
                        ax.set_title('Hasil Prediksi Batch', fontsize=14, fontweight='bold', pad=20)
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}',
                                   ha='center', va='bottom', fontsize=11, fontweight='bold')
                        
                        st.pyplot(fig)
                        plt.close()
        
        except Exception as e:
            st.error(f"Error memproses file: {str(e)}")
    
    else:
        st.info("Silakan upload file CSV untuk memulai prediksi batch.")
        
        # Sample CSV format
        st.markdown("### Format CSV yang Benar")
        st.markdown("""
        File CSV Anda harus berbentuk seperti ini:
        
        ```
        message
        "Congratulations! You won a prize. Call now!"
        "Hey, are you coming to the party tonight?"
        "URGENT: Your account needs verification. Click here."
        "Thanks for the update. See you tomorrow!"
        ```
        """)


# ==================== PAGE 6: ABOUT ====================

elif page == "About":
    st.title("Tentang Aplikasi Ini")
    st.markdown("---")
    
    st.markdown("""
    ### Sistem Deteksi Spam SMS
    
    Aplikasi ini adalah sistem deteksi spam SMS yang komprehensif yang menggunakan 
    **beberapa algoritma machine learning** untuk mengklasifikasikan pesan teks 
    secara akurat sebagai spam atau pesan legitimate (ham).
    """)
    
    st.markdown("### Informasi Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Mahasiswa:** Wahyu Pratama  
        **NIM:** 230411100058  
        **Mata Kuliah:** Pemrosesan Bahasa Alami (NLP)  
        **Project:** Tugas UAS - SMS Spam Detection  
        **Tahun:** 2026
        """)
    
    with col2:
        st.markdown("""
        **Ukuran Dataset:** 5,572 pesan SMS  
        **Jumlah Model:** 4 algoritma Machine Learning  
        **Rentang Akurasi:** 95% - 98%  
        **Teknologi:** Python, Streamlit, Scikit-learn
        """)
    
    st.markdown("### Model Machine Learning")
    
    models_description = {
        "Logistic Regression": {
            "description": "Algoritma klasifikasi linear yang menggunakan fungsi logistik untuk memodelkan hasil binary.",
            "pros": "Cepat, mudah diinterpretasi, bekerja baik dengan data yang dapat dipisahkan secara linear",
            "use_case": "Model baseline, prediksi real-time"
        },
        "Support Vector Machine (SVM)": {
            "description": "Mencari hyperplane optimal yang memisahkan kelas yang berbeda dalam ruang berdimensi tinggi.",
            "pros": "Efektif di ruang berdimensi tinggi, efisien dalam penggunaan memori",
            "use_case": "Klasifikasi teks, pengenalan pola"
        },
        "Random Forest": {
            "description": "Metode ensemble yang membangun banyak decision trees dan menggabungkan prediksi mereka.",
            "pros": "Menangani hubungan non-linear, robust terhadap overfitting",
            "use_case": "Dataset kompleks, analisis feature importance"
        },
        "Multinomial Naive Bayes": {
            "description": "Classifier probabilistik berdasarkan teorema Bayes dengan asumsi independensi naive.",
            "pros": "Training cepat, bekerja baik dengan data teks, output probabilistik",
            "use_case": "Klasifikasi teks, filtering spam"
        }
    }
    
    for model_name, info in models_description.items():
        with st.expander(f"{model_name}"):
            st.markdown(f"**Deskripsi:** {info['description']}")
            st.markdown(f"**Keunggulan:** {info['pros']}")
            st.markdown(f"**Kasus Penggunaan:** {info['use_case']}")
    
    st.markdown("### Cara Kerja")
    
    st.markdown("""
    #### Pipeline Pemrosesan:
    
    1. **Input Teks**  
       User memasukkan atau mengupload pesan SMS
    
    2. **Preprocessing**
       - Konversi ke lowercase
       - Hapus karakter khusus dan tanda baca
       - Tokenisasi menjadi kata-kata
       - Hapus stopwords (kata umum seperti 'the', 'is', 'a')
       - Lemmatization (konversi kata ke bentuk dasar: 'running' → 'run')
    
    3. **Ekstraksi Fitur**
       - Vektorisasi TF-IDF (Term Frequency-Inverse Document Frequency)
       - Mengkonversi teks menjadi fitur numerik
       - Menonjolkan kata-kata penting sambil mengurangi bobot kata umum
    
    4. **Prediksi Model**
       - Fitur yang diproses dimasukkan ke model ML yang dipilih
       - Output model: Klasifikasi Spam atau Ham
       - Confidence score (probabilitas) jika tersedia
    
    5. **Tampilan Hasil**
       - Visualisasi prediksi yang jelas
       - Metrik confidence
       - Indikator spam yang teridentifikasi
       - Detail pemrosesan
    """)
    
    st.markdown("### Informasi Dataset")
    
    st.markdown("""
    Model dilatih menggunakan **SMS Spam Collection Dataset** yang berisi:
    
    - **Total Pesan:** 5,572 pesan SMS
    - **Pesan Ham:** 4,825 (86.6%) - Pesan legitimate
    - **Pesan Spam:** 747 (13.4%) - Pesan spam/tidak diinginkan
    - **Sumber:** Pesan SMS nyata yang dikumpulkan dari berbagai sumber
    - **Format:** File CSV dengan kolom label dan message
    - **Bahasa:** Terutama Bahasa Inggris
    """)

    
    st.markdown("### Fitur-Fitur")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Prediksi Real-time**  
        Klasifikasi instan untuk pesan SMS individual
        
        **Multiple Models**  
        Pilih dari 4 algoritma ML yang berbeda
        
        **Batch Processing**  
        Upload file CSV untuk prediksi massal
        
        **Metrik Performa**  
        Detail akurasi, presisi, recall, dan F1-score
        """)
    
    with col2:
        st.markdown("""
        **Visualisasi Data**  
        Word clouds, grafik, dan analitik
        
        **Indikator Spam**  
        Identifikasi pola mencurigakan dalam pesan
        
        **Export Hasil**  
        Download hasil prediksi dalam format CSV
        
        **Interface Interaktif**  
        Antarmuka web yang modern dan responsif
        """)
    
    st.markdown("### Teknologi yang Digunakan")
    
    st.markdown("""
    - **Python 3.x** - Bahasa pemrograman utama
    - **Streamlit** - Framework aplikasi web
    - **Scikit-learn** - Library machine learning
    - **Pandas** - Manipulasi dan analisis data
    - **NLTK** - Toolkit pemrosesan bahasa alami
    - **Matplotlib & Seaborn** - Visualisasi data
    - **WordCloud** - Pembuatan word cloud
    - **Joblib** - Serialisasi model
    """)
    
    st.markdown("### Referensi")
    
    st.markdown("""
    - UCI Machine Learning Repository - SMS Spam Collection Dataset
    - Dokumentasi Scikit-learn
    - Natural Language Processing with Python (NLTK)
    - Dokumentasi Streamlit
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>SMS Spam Detection System</strong></p>
        <p>Dikembangkan untuk Tugas UAS - Pemrosesan Bahasa Alami</p>
        <p>© 2026 - NIM: 230411100058</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9rem; padding: 1rem;'>
    Wahyu Pratama (230411100058) | NLP Project 2026
</div>
""", unsafe_allow_html=True)
