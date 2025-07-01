import os

# Cấu trúc thư mục

folders = [
    "data",
    "notebooks",
    "src",
    "app",
    "models"
]

files = {
    "README.md"
    ".gitignore": "*.pyc\n__pycache__/\nmodels/\n.env\n",
    "requirements.txt": "pandas\nscikit-learn\nnumpy\nstreamlit\njoblib\nmatplotlib\nseaborn\n",
    "main.py": "# Main pipeline entry (optional)",
    "notebooks/EDA_and_Modeling.ipynb": "",
    "src/preprocess.py": "# Functions for preprocessing data",
    "src/train_model.py": "# Functions for training model",
    "src/evaluate.py": "# Functions for evaluating model",
    "src/predict.py": "# Functions for making predictions",
    "app/app.py": "# Streamlit app entrypoint",
    "app/model_utils.py": "# Functions to load model and scaler"
}
# Tạo thư mục con
for folder in folders:
    os.makedirs(os.path.join(folder), exist_ok=True)

# Tạo file
for filepath, content in files.items():
    full_path = os.path.join( filepath)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

print(f"✅ Project template has been created!")
