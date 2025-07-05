import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # 1. Tách cột dạng số & dạng phân loại
    numerical_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome', 'PercentSalaryHike']
    categorical_cols = ['BusinessTravel', 'JobRole', 'MaritalStatus', 'OverTime']

     # 2. Tách feature + label
    X = df[numerical_cols + categorical_cols]
    y = df['Attrition'].map({'No': 0, 'Yes': 1})

    # 3. Scale
    scaler = MinMaxScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    joblib.dump(scaler, 'models/scaler.pkl')

    # 4. Encode
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X[categorical_cols])
    joblib.dump(encoder, 'models/encoder.pkl')

    # 5. Biến encode + merge
    def encode_and_merge(X):
        X = X.reset_index(drop=True)
        encoded = pd.DataFrame(
            encoder.transform(X[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X.index
        )
        return pd.concat([X.drop(columns=categorical_cols), encoded], axis=1)

    X = encode_and_merge(X)
    return X, y, encoder, scaler
