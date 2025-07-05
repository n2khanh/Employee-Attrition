
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Thêm src vào path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import prepare_input

st.set_page_config(page_title="Attrition Predictor", layout="centered")
st.title("🧠 Employee Attrition Predictor")
st.markdown("Dự đoán khả năng nghỉ việc của nhân viên.")

# ===== Load model và pipeline =====
@st.cache_resource
def load_assets():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_assets()

# ===== Định nghĩa feature =====
numerical_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome', 'PercentSalaryHike']
categorical_cols = ['BusinessTravel', 'JobRole', 'MaritalStatus', 'OverTime']
category_dict = dict(zip(categorical_cols, encoder.categories_))

# ===== Giao diện =====
with st.form("attrition_form"):
    # Nhập dữ liệu dạng số
    age = st.slider("Tuổi", 18, 60, 30)
    income = st.number_input("Thu nhập hàng tháng", min_value=1000, max_value=30000, value=5000, step=500)
    years = st.slider("Số năm làm việc tại công ty", 0, 40, 5)
    distance = st.slider("Khoảng cách từ nhà đến công ty (km)", 1, 50, 10)
    hike = st.slider("Tăng lương gần nhất (%)", 0, 100, 15)

    # Nhập dữ liệu phân loại
    travel = st.selectbox("Tần suất đi công tác", category_dict['BusinessTravel'])
    jobrole = st.selectbox("Vị trí công việc", category_dict['JobRole'])
    marital = st.selectbox("Tình trạng hôn nhân", category_dict['MaritalStatus'])
    overtime = st.selectbox("Làm thêm giờ", category_dict['OverTime'])

    submit = st.form_submit_button("🔮 Dự đoán")

if submit:
    input_data = {
        "Age": age,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "DistanceFromHome": distance,
        "PercentSalaryHike": hike,
        "BusinessTravel": travel,
        "JobRole": jobrole,
        "MaritalStatus": marital,
        "OverTime": overtime,
    }

    # Kiểm tra đủ feature
    missing = set(numerical_cols + categorical_cols) - set(input_data.keys())
    if missing:
        st.error(f"❌ Thiếu các feature: {missing}")
    else:
        try:
            X_input = prepare_input(input_data, numerical_cols, categorical_cols, scaler, encoder)
            proba = model.predict_proba(X_input)[0][1]
            label = "🔴 Có khả năng nghỉ việc" if proba > 0.5 else "🟢 Ở lại công ty"

            st.markdown("---")
            st.subheader("📊 Kết quả dự đoán:")
            st.write(f"**{label}**  Xác suất nghỉ việc: **{proba:.2%}**")

        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
