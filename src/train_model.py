from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

def train_model(X, y, save_path="models/best_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1",  
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)
    joblib.dump(grid_search.best_estimator_, save_path)
    return grid_search.best_estimator_, X_test, y_test
