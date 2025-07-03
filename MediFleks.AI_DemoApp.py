# ==================== Diagnosa Epilepsi ==============================#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

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
url = 'https://raw.githubusercontent.com/BDCIMERIGit/MediFleks.AI.Demo/main/dataset_epilepsi_anak.xlsx'  # Update this with the actual URL
response = requests.get(url)
with open('dataset_epilepsi_anak.xlsx', 'wb') as f:
    f.write(response.content)

# Load the dataset
df = pd.read_excel("dataset_epilepsi_anak.xlsx")

df.info()

# Encode label dan kategorikal
le = LabelEncoder()
for col in ['Jenis_Kelamin', 'Jumlah_Obat', 'Hasil_EEG', 'Hasil_MRI_Kepala', 'Penurunan_Frekuensi_Kejang']:
    df[col] = le.fit_transform(df[col])

# Target encoding
df['Jenis Epilepsi (Target)'] = le.fit_transform(df['Jenis Epilepsi (Target)'])
target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Split into features (X) and target (y)
X = df.drop(columns=['Jenis_Epilepsi'])  # Assuming 'target' is the label column
y = df['Jenis_Epilepsi']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_model(model, params, model_name):
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    print(f"Best Parameters for {model_name}:", grid.best_params_)
    return grid.best_estimator_

models = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "DecisionTree": (DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}),
    "RandomForest": (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
    "SVM": (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    "XGBoost": (XGBClassifier(eval_metric='mlogloss'), {'n_estimators': [50, 100], 'max_depth': [3, 5]})
}

trained_models = {}
for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    trained_models[name] = train_model(model, params, name)
    
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred, target_names=target_mapping.keys()))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=target_mapping.keys())
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # ROC Curve
    if y_proba.shape[1] == 3:
        fpr = {}
        tpr = {}
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
            plt.plot(fpr[i], tpr[i], label=f"{list(target_mapping.keys())[i]}")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid()
        plt.show()

for name, model in trained_models.items():
    evaluate_model(model, name)

from sklearn.model_selection import learning_curve

def plot_learning_curve(model, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_scores_mean = train_scores.mean(axis=1)
    val_scores_mean = val_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
    plt.plot(train_sizes, val_scores_mean, label='Validation Accuracy')
    plt.title(f'Learning Curve - {title}')
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# Contoh untuk Random Forest
plot_learning_curve(trained_models['LogisticRegression'], "Logistic Regression")
plot_learning_curve(trained_models['KNN'], "KNN")
plot_learning_curve(trained_models['DecisionTree'], "Decision Tree")
plot_learning_curve(trained_models['RandomForest'], "Random Forest")
plot_learning_curve(trained_models['SVM'], "SVM")
plot_learning_curve(trained_models['XGBoost'], "XG Boost")

from sklearn.metrics import accuracy_score

# Hitung akurasi semua model
accuracy_scores = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc

# Plot akurasi
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.ylim(0, 1)
plt.ylabel("Accuracy Score")
plt.title("Perbandingan Akurasi Model untuk Klasifikasi Epilepsi")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

import joblib

# Simpan model terbaik (misalnya kita pilih Random Forest, atau bisa ganti ke model lain sesuai evaluasi)
final_model = trained_models['DecisionTree']  # ganti sesuai model terbaikmu
joblib.dump(final_model, 'ModelDiagnosaEpilepsi.pkl')
print("Model saved as ModelDiagnosaEpilepsi.pkl")

# ================ Diagnosa Diabetes ======================= #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
url = 'https://raw.githubusercontent.com/BDCIMERIGit/MediFleks.AI.Demo/main/dummy_diabetes_8000.xlsx'  # Update this with the actual URL
response = requests.get(url)
with open('dataset_epilepsi_anak.xlsx', 'wb') as f:
    f.write(response.content)
    
df = pd.read_excel("dummy_diabetes_8000.xlsx")

# Target mapping
target_mapping = {
    0: "Normal",
    1: "Diabetes"
}

# Split into features (X) and target (y)
X = df.drop(columns=['Outcome', 'Pregnancies'])  # Assuming 'target' is the label column
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def train_model(model, params, model_name):
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    print(f"Best Parameters for {model_name}:", grid.best_params_)
    return grid.best_estimator_

models = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    "DecisionTree": (DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}),
    "RandomForest": (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
    "SVM": (SVC(probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    "XGBoost": (XGBClassifier(eval_metric='mlogloss'), {'n_estimators': [50, 100], 'max_depth': [3, 5]})
}

trained_models = {}
for name, (model, params) in models.items():
    print(f"\nTraining {name}...")
    trained_models[name] = train_model(model, params, name)
    
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Fix: convert int -> label string
    target_names = [target_mapping[i] for i in sorted(target_mapping.keys())]

    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)
        n_classes = y_proba.shape[1]
        for i in range(n_classes):
            if np.any(y_test == i):
                fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
                plt.plot(fpr, tpr, label=target_names[i])

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print(f"{model_name} does not support predict_proba(), skipping ROC Curve.")

for name, model in trained_models.items():
    evaluate_model(model, name)
    
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_scaled, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_scores_mean = train_scores.mean(axis=1)
    val_scores_mean = val_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
    plt.plot(train_sizes, val_scores_mean, label='Validation Accuracy')
    plt.title(f'Learning Curve - {title}')
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

# Contoh untuk Random Forest
plot_learning_curve(trained_models['LogisticRegression'], "Logistic Regression")
plot_learning_curve(trained_models['KNN'], "KNN")
plot_learning_curve(trained_models['DecisionTree'], "Decision Tree")
plot_learning_curve(trained_models['RandomForest'], "Random Forest")
plot_learning_curve(trained_models['SVM'], "SVM")
plot_learning_curve(trained_models['XGBoost'], "XG Boost")

from sklearn.metrics import accuracy_score

# Hitung akurasi semua model
accuracy_scores = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc

# Plot akurasi
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.ylim(0, 1)
plt.ylabel("Accuracy Score")
plt.title("Perbandingan Akurasi Model untuk Klasifikasi Epilepsi")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

import joblib

# Simpan model terbaik (misalnya kita pilih Random Forest, atau bisa ganti ke model lain sesuai evaluasi)
final_model = trained_models['DecisionTree']  # ganti sesuai model terbaikmu
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
with open('dataset_epilepsi_anak.xlsx', 'wb') as f:
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

# Load saved models
model_epilepsi = joblib.load("ModelDiagnosaEpilepsi.pkl")
model_diabetes = joblib.load("ModelDiagnosaDiabetes.pkl")
model_jantung = joblib.load("ModelDiagnosaSeranganJantung.pkl")

# --- Login Page ---
def login_page():
    st.title("Selamat Datang di MediFleks.AI")
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

# --- Choose Disease Page ---
def choose_disease():
    st.title("Choose Disease")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Epilepsi")
        if st.button("Start Epilepsi"):
            st.session_state.page = "epilepsi"

    with col2:
        st.subheader("Diabetes")
        if st.button("Start Diabetes"):
            st.session_state.page = "diabetes"

    with col3:
        st.subheader("Serangan Jantung")
        if st.button("Start Jantung"):
            st.session_state.page = "jantung"

# --- Epilepsi Diagnosis ---
def diagnose_epilepsi():
    st.title("Form Diagnosa Epilepsi")
    jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    usia = st.number_input("Usia", min_value=1, max_value=100, value=10)
    obat = st.selectbox("Jumlah Obat", [1, 2, 3])
    eeg = st.selectbox("Hasil EEG", ["Normal", "Sindrom epilepsi", "Abnormal dengan gelombang epileptiform"])
    mri = st.selectbox("Hasil MRI Kepala", ["Normal", "Abnormal Epileptogenik", "Abnormal non-epileptogenik"])
    penurunan = st.selectbox("Penurunan Frekuensi Kejang", ["Ya", "Tidak"])

    if st.button("Start Diagnosa"):
        input_data = np.array([[jk, usia, obat, eeg, mri, penurunan]])
        prediction = model_epilepsi.predict(input_data)[0]
        st.success(f"Hasil Diagnosa: {prediction}")

        if st.button("Simpan hasil diagnosis"):
            st.success("Hasil diagnosis tersimpan")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Diagnosis penyakit lain"):
                    st.session_state.page = "choose_disease"
            with col2:
                if st.button("Keluar dari aplikasi"):
                    st.session_state.page = "login"

# --- Diabetes Diagnosis ---
def diagnose_diabetes():
    st.title("Form Diagnosa Diabetes")
    glucose = st.number_input("Glucose", 70, 200)
    bp = st.number_input("Blood Pressure", 55, 160)
    skin = st.number_input("Skin Thickness", 10, 50)
    insulin = st.number_input("Insulin", 30, 280)
    bmi = st.number_input("BMI", 18.0, 45.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.1, 3.0)
    age = st.number_input("Umur", 20, 80)

    if st.button("Start Diagnosa"):
        input_data = np.array([[glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = model_diabetes.predict(input_data)[0]
        st.success(f"Hasil Diagnosa: {'Positif Diabetes' if prediction == 1 else 'Negatif Diabetes'}")

        if st.button("Simpan hasil diagnosis"):
            st.success("Hasil diagnosis tersimpan")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Diagnosis penyakit lain"):
                    st.session_state.page = "choose_disease"
            with col2:
                if st.button("Keluar dari aplikasi"):
                    st.session_state.page = "login"

# --- Heart Disease Diagnosis ---
def diagnose_jantung():
    st.title("Form Diagnosa Serangan Jantung")
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
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])
        prediction = model_jantung.predict(input_data)[0]
        st.success(f"Hasil Diagnosa: {'High Risk' if prediction == 1 else 'Low Risk'}")

        if st.button("Simpan hasil diagnosis"):
            st.success("Hasil diagnosis tersimpan")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Diagnosis penyakit lain"):
                    st.session_state.page = "choose_disease"
            with col2:
                if st.button("Keluar dari aplikasi"):
                    st.session_state.page = "login"

# --- Routing Halaman ---
if "page" not in st.session_state:
    st.session_state.page = "home"

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
