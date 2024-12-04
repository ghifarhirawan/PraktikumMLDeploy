import streamlit as st
import joblib
import numpy as np

# Load model K-Means
model = joblib.load('kmeans_model.pkl')

# Judul aplikasi
st.title("Prediksi Cluster dengan K-Means")
st.write("Gunakan slider di bawah untuk memasukkan data baru dan prediksi cluster.")

# Input fitur dengan slider
st.header("Input Data Baru:")
sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1, value=5.0)
sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, step=0.1, value=3.5)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1, value=1.4)
petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, step=0.1, value=0.2)

# Tombol prediksi
if st.button("Prediksi"):
    # Format input untuk model
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Prediksi cluster
    predicted_cluster = model.predict(new_data)
    
    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(f"Data baru masuk ke cluster: *{predicted_cluster[0]}*")