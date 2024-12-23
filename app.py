import streamlit as st
import pandas as pd
import joblib

kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

st.title('User Behavior Clustering App')

with st.form(key='user_input_form'):
    st.subheader("Input User Data")

    app_usage = st.number_input("App Usage Time (min/day)", min_value=0, max_value=1440, step=1)
    screen_on_time = st.number_input("Screen On Time (hours/day)", min_value=0.0, max_value=24.0, step=0.1)
    battery_drain = st.number_input("Battery Drain (mAh/day)", min_value=0, max_value=5000, step=10)
    apps_installed = st.number_input("Number of Apps Installed", min_value=0, max_value=500, step=1)
    data_usage = st.number_input("Data Usage (MB/day)", min_value=0.0, max_value=10000.0, step=0.1)
    age = st.number_input("Age", min_value=18, max_value=100, step=1)

    submit_button = st.form_submit_button(label='Predict Cluster')

if submit_button:
    try:
        input_data = [
            app_usage,
            screen_on_time,
            battery_drain,
            apps_installed,
            data_usage,
            age
        ]
        input_df = pd.DataFrame([input_data], columns=[
            'App Usage Time (min/day)', 'Screen On Time (hours/day)',
            'Battery Drain (mAh/day)', 'Number of Apps Installed',
            'Data Usage (MB/day)', 'Age'
        ])

        scaled_input = scaler.transform(input_df)

        pca_input = pca.transform(scaled_input)

        cluster = kmeans.predict(pca_input)
        cluster_label = cluster[0]

        st.success(f"The predicted cluster is: **{cluster_label}**")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
