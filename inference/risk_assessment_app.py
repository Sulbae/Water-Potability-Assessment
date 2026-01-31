import streamlit as st
import pandas as pd
import time
from joblib import load

st.set_page_config(
    page_title="Aplikasi Risk Assessment Monitoring Kelayakan Air",
    page_icon="💧",
    layout="centered",
)

THRESHOLD_POTABILITY = 0.69

# Load Artifacts
def load_artifacts():
    try:
        preprocess = load("artifacts/preprocessing_pipeline.pkl")
        clf_model  = load("artifacts/rf_classifier_model.pkl")
        anom_model = load("artifacts/anomaly_detection_model.pkl")
        return preprocess, clf_model, anom_model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

PREPROCESSOR, CLF_MODEL, ANOM_MODEL = load_artifacts()

def run_inferece(data_input: pd.DataFrame) -> dict:
    # preprocessing
    X = PREPROCESSOR.transform(data_input)

    # classification
    potability_probs = CLF_MODEL.predict_proba(X)[0, 1]
    potability_pred = int(potability_probs >= THRESHOLD_POTABILITY)

    # anomaly detection
    anom_flag = ANOM_MODEL.predict(X)[0] # 1: normal, -1: anomali
    is_anom = int(anom_flag == -1) # 0: normal, 1: anomali

    # risk assessment
    if potability_pred == 1 and is_anom == 0:
        risk_label = "AMAN"
        recommendation = (
            "Air layak konsumsi dan sesuai pola normal." 
            "Dapat didistribusikan dengan monitoring rutin."
        )
    
    elif potability_pred == 1 and is_anom == 1:
        risk_label = "PERLU DICEK"
        recommendation = (
            "Air layak konsumsi, namun polanya tidak umum."
            "Disarankan untuk melakukan pengecekan lebih lanjut."
        )
    
    elif potability_pred == 0 and is_anom == 0:
        risk_label = "TIDAK LAYAK"
        recommendation = (
            "Air tidak layak konsumsi dan pola normal."
            "Perlu optimasi proses pengolahan air dan pengecekan lebih lanjut."
        )
    else: # potability_pred == 0 and is_anom == 1:
        risk_label = "KRITIS"
        recommendation = (
            "Air sangat tidak layak konsumsi dan terdeteksi anomali ekstrim."
            "Wajib verifikasi laboratorium dan penanganan darurat."
        )
    return {
        "risk_label": risk_label,
        "recommendation": recommendation
    }

# Streamlit UI
st.title("Aplikasi Risk Assessment Monitoring Kelayakan Air 💧")
st.markdown("Masukkan data setiap parameter kualitas air untuk melakukan Risk Assessment Kelayakan Air**!")

# Cek status model
if CLF_MODEL is not None and ANOM_MODEL is not None:
    st.caption(f"Status: Model sudah siap.")
else:
    st.error("MODEL GAGAL DIMUAT. Periksa folder artifacts.")
    st.stop()

# Input Form
st.subheader("Input Data Parameter Kualitas Air")

columns = [
        "ph", "Hardness", "Solids", 
        "Chloramines", "Sulfate", "Conductivity", 
        "Organic_carbon", "Trihalomethanes", "Turbidity"
]

## Input data numerik
ph = st.number_input("ph", min_value=0.1, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.1, max_value=1000.0, value=200.0)
solids = st.number_input("Solids", min_value=0.1, max_value=100000.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.1, max_value=100.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.1, max_value=1000.0, value=300.0)
conductivity = st.number_input("Conductivity", min_value=0.1, max_value=1000.0, value=400.0)
organic_carbon = st.number_input("Organic_carbon", min_value=0.1, max_value=1000.0, value=15.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.1, max_value=1000.0, value=80.0)
turbidity = st.number_input("Turbidity", min_value=0.1, max_value=100.0, value=4.0)

# Menyimpan data ke dalam DataFrame
data_input = pd.DataFrame([[
    ph, hardness, solids, 
    chloramines, sulfate, conductivity, 
    organic_carbon, trihalomethanes, turbidity
]], columns=columns)

# Tombol untuk menampilkan data
if st.button("Prediksi Kelayakan Air", type="primary"):

    start_time = time.time()

    st.write("### Data Input:")
    st.dataframe(data_input)
  
    try:
        result = run_inferece(data_input)

        st.write("### Hasil")

        # Risk Level
        st.subheader("Risk Level dan Rekomendasi:")

        risk_label = result["risk_label"]
        recommendation = result["recommendation"]

        st.write(f"**Risk Label:** {risk_label}")
        st.write(f"**Rekomendasi:** {recommendation}")

    except Exception as e:
        st.error(f"Terjadi kesalahan sistem: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.caption(f"Waktu inferensi: {elapsed_time:.2f} detik")