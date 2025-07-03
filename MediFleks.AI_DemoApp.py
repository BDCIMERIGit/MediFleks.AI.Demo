# ==================== Diagnosa Epilepsi ==============================#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Utils
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import warnings
warnings.filterwarnings('ignore')

# Load the dataset from GitHub
url = 'https://raw.githubusercontent.com/BDCIMERIGit/MediFleks.AI.Demo/main/dataset_epilepsi_anak.xlsx'
response = requests.get(url)
with open('dataset_epilepsi_anak.xlsx', 'wb') as f:
    f.write(response.content)

# Load the dataset
df = pd.read_excel("dataset_epilepsi_anak.xlsx")

# Encode label dan kategorikal
le = LabelEncoder()
for col in ['Jenis_Kelamin', 'Jumlah_Obat', 'Hasil_EEG', 'Hasil_MRI_Kepala', 'Penurunan_Frekuensi_Kejang']:
    df[col] = le.fit_transform(df[col])

df['Jenis_Epilepsi'] = le.fit_transform(df['Jenis_Epilepsi'])

# Split into features and target
X = df.drop(columns=['Jenis_Epilepsi'])
y = df['Jenis_Epilepsi']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree only
model = DecisionTreeClassifier()
params = {'max_depth': [3, 5, 10]}
grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
final_model = grid.best_estimator_

# Simpan model
joblib.dump(final_model, 'ModelDiagnosaEpilepsi.pkl')
print("Model saved as ModelDiagnosaEpilepsi.pkl")

# ================ Diagnosa Diabetes ======================= #

# Load the dataset from GitHub
url = 'https://raw.githubusercontent.com/BDCIMERIGit/MediFleks.AI.Demo/main/dummy_diabetes_8000.xlsx'
response = requests.get(url)
with open('dummy_diabetes_8000.xlsx', 'wb') as f:
    f.write(response.content)

df = pd.read_excel("dummy_diabetes_8000.xlsx")

# Split into features and target
X = df.drop(columns=['Outcome', 'Pregnancies'])
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree only
model = DecisionTreeClassifier()
params = {'max_depth': [3, 5, 10]}
grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
final_model = grid.best_estimator_

# Simpan model
joblib.dump(final_model, 'ModelDiagnosaDiabetes.pkl')
print("Model saved as ModelDiagnosaDiabetes.pkl")

# ================ Diagnosa Serangan Jantung =============== #

#Test Deploy Heart Disease App Streamlit

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset from GitHub
url = 'https://raw.githubusercontent.com/BDCIMERIGit/MediFleks.AI.Demo/main/heart.csv'  # Update this with the actual URL
response = requests.get(url)
with open('heart.csv', 'wb') as f:
    f.write(response.content)

df = pd.read_csv("heart.csv")

# Split into features (X) and target (y)
X = df.drop(columns=['target'])  # Assuming 'target' is the label column
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with SVM model
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(C=1, gamma='scale', kernel='linear'))
])

# Train the model
svm_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = svm_pipeline.predict(X_test)

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save the model
joblib.dump(svm_pipeline, 'ModelDiagnosaSeranganJantung.pkl')
print("Model saved as 'ModelDiagnosaSeranganJantung.pkl'")

# ================ Aplikasi Streamlit ====================== #

import streamlit as st
import numpy as np
import joblib

# ============ Custom CSS Styling ============ #
st.markdown(
    """
    <style>
    .stApp {
        background-color: #00bf63;
    }
    h1, h2, h3, h4 {
        color: white;
    }
    .stMarkdown, .stTextInput > label, .stNumberInput > label, .stSelectbox > label {
        color: white !important;
    }
    div.stButton > button {
        background-color: #ffffff;
        color: #00bf63;
        border-radius: 8px;
        padding: 0.5em 1.2em;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #f0f0f0;
        color: #007c47;
        transform: scale(1.02);
    }
    .stTextInput > div > input,
    .stNumberInput input,
    .stSelectbox > div > div {
        background-color: #e6ffe6;
        color: black;
        border-radius: 6px;
    }
    .center-title {
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    .diagnosis-box {
        background-color: #ffffff;
        padding: 15px 20px;
        border-radius: 12px;
        margin-top: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }
    .diagnosis-box h3 {
        color: #00bf63;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load saved models
model_epilepsi = joblib.load("ModelDiagnosaEpilepsi.pkl")
model_diabetes = joblib.load("ModelDiagnosaDiabetes.pkl")
model_jantung = joblib.load("ModelDiagnosaSeranganJantung.pkl")

label_mapping = {
    'Jenis_Kelamin': {'Laki-laki': 1, 'Perempuan': 0},
    'Jumlah_Obat': {1: 0, 2: 1, 3: 2},
    'Hasil_EEG': {'Normal': 1, 'Sindrom epilepsi': 2, 'Abnormal dengan gelombang epileptiform': 0},
    'Hasil_MRI_Kepala': {'Normal': 2, 'Abnormal Epileptogenik': 0, 'Abnormal non-epileptogenik': 1},
    'Penurunan_Frekuensi_Kejang': {'Ya': 1, 'Tidak': 0}
}

epilepsi_labels = {0: "Epilepsi Fokal", 1: "Epilepsi Umum", 2: "Sindrom Epilepsi"}

# --- Routing Halaman ---
if "page" not in st.session_state:
    st.session_state.page = "home"

# --- Halaman-halaman ---

def login_page():
    st.markdown("<h1 class='center-title'>Selamat Datang di MediFleks.AI</h1>", unsafe_allow_html=True)
    if st.button("Start App"):
        st.session_state.page = "login"

def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "AdminMediFleks" and password == "admin123":
            st.session_state.page = "choose_disease"
        else:
            st.error("Username atau Password salah")

def choose_disease():
    st.markdown("<h2 class='center-title'>Pilih Jenis Diagnosa</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧠 Epilepsi"):
            st.session_state.page = "epilepsi"
    with col2:
        if st.button("💉 Diabetes"):
            st.session_state.page = "diabetes"
    with col3:
        if st.button("❤️ Serangan Jantung"):
            st.session_state.page = "jantung"

def diagnose_epilepsi():
    st.markdown("<h2 class='center-title'>⚡ Diagnosa Epilepsi</h2>", unsafe_allow_html=True)
    jk = st.selectbox("Jenis Kelamin", list(label_mapping['Jenis_Kelamin'].keys()))
    usia = st.number_input("Usia", min_value=1, max_value=100, value=10)
    obat = st.selectbox("Jumlah Obat", list(label_mapping['Jumlah_Obat'].keys()))
    eeg = st.selectbox("Hasil EEG", list(label_mapping['Hasil_EEG'].keys()))
    mri = st.selectbox("Hasil MRI Kepala", list(label_mapping['Hasil_MRI_Kepala'].keys()))
    penurunan = st.selectbox("Penurunan Frekuensi Kejang", list(label_mapping['Penurunan_Frekuensi_Kejang'].keys()))

    if st.button("Start Diagnosa"):
        input_data = np.array([[
            label_mapping['Jenis_Kelamin'][jk], usia,
            label_mapping['Jumlah_Obat'][obat],
            label_mapping['Hasil_EEG'][eeg],
            label_mapping['Hasil_MRI_Kepala'][mri],
            label_mapping['Penurunan_Frekuensi_Kejang'][penurunan]
        ]])
        pred = model_epilepsi.predict(input_data)[0]
        hasil = epilepsi_labels.get(pred, "Tidak diketahui")
        st.markdown(f"<div class='diagnosis-box'><h3>Hasil Diagnosa: {hasil}</h3></div>", unsafe_allow_html=True)
        if st.button("Simpan hasil diagnosis"):
            st.success("Hasil diagnosis tersimpan")
            st.session_state.page = "choose_disease"

def diagnose_diabetes():
    st.markdown("<h2 class='center-title'>💉 Diagnosa Diabetes</h2>", unsafe_allow_html=True)
    glucose = st.number_input("Glucose", 70, 200)
    bp = st.number_input("Blood Pressure", 55, 160)
    skin = st.number_input("Skin Thickness", 10, 50)
    insulin = st.number_input("Insulin", 30, 280)
    bmi = st.number_input("BMI", 18.0, 45.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.1, 3.0)
    age = st.number_input("Umur", 20, 80)
    if st.button("Start Diagnosa"):
        input_data = np.array([[glucose, bp, skin, insulin, bmi, dpf, age]])
        pred = model_diabetes.predict(input_data)[0]
        hasil = 'Positif Diabetes' if pred == 1 else 'Negatif Diabetes'
        st.markdown(f"<div class='diagnosis-box'><h3>Hasil Diagnosa: {hasil}</h3></div>", unsafe_allow_html=True)
        if st.button("Simpan hasil diagnosis"):
            st.success("Hasil diagnosis tersimpan")
            st.session_state.page = "choose_disease"

def diagnose_jantung():
    st.markdown("<h2 class='center-title'>❤️ Diagnosa Serangan Jantung</h2>", unsafe_allow_html=True)
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.slider("Chest Pain Type (CP)", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", 50, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.slider("Resting ECG", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.slider("Slope of ST Segment", 0, 2, 1)
    ca = st.slider("Major Vessels Colored", 0, 3, 0)
    thal = st.slider("Thalassemia Type", 0, 2, 1)

    if st.button("Start Diagnosa"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        pred = model_jantung.predict(input_data)[0]
        hasil = 'High Risk' if pred == 1 else 'Low Risk'
        st.markdown(f"<div class='diagnosis-box'><h3>Hasil Diagnosa: {hasil}</h3></div>", unsafe_allow_html=True)
        if st.button("Simpan hasil diagnosis"):
            st.success("Hasil diagnosis tersimpan")
            st.session_state.page = "choose_disease"

if st.session_state.page == "home":
    login_page()
elif st.session_state.page == "login":
    login()
elif st.session_state.page == "choose_disease":
    choose_disease()
elif st.session_state.page == "epilepsi":
    diagnose_epilepsi()
elif st.session_state.page == "diabetes":
    diagnose_diabetes()
elif st.session_state.page == "jantung":
    diagnose_jantung()
