import joblib
import numpy as np
import pandas as pd

def load_model_and_encoder(model_path="models/best_model.pkl", scaler_path="models/scaler.pkl", encoder_path="models/encoder.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    return model, scaler, encoder

def prepare_input(user_input: dict, numerical_cols, categorical_cols, scaler, encoder):
    """
    user_input: dict từ web hoặc API gửi lên
    Trả về 1 sample đã scale + encode sẵn để predict
    """
    # Convert dict → DataFrame
    input_df = pd.DataFrame([user_input])

    # Scale numerical
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Encode categorical
    encoded = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Ghép lại
    input_df.reset_index(drop=True, inplace=True)
    final_df = pd.concat([input_df.drop(columns=categorical_cols), encoded_df], axis=1)
    return final_df

def predict_attrition(user_input: dict, model, scaler, encoder, numerical_cols, categorical_cols):
    """
    Trả về xác suất và nhãn dự đoán (Yes/No)
    """
    processed_input = prepare_input(user_input, numerical_cols, categorical_cols, scaler, encoder)
    proba = model.predict_proba(processed_input)[0][1]  # xác suất nghỉ việc (Attrition=Yes)
    label = "Yes" if proba > 0.5 else "No"
    return label, proba
