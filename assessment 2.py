"""
CAR PRICE PREDICTION
Linear Regression vs K-Nearest Neighbors Regression

Author: Fabiola Zeth Patero
Student ID: 2023-191

Note:
AI tools were used only for general assistance (e.g., debugging support), similar to using documentation or online references.
All modelling choices, analysis logic, and final decisions were made by the author.
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance

np.random.seed(42)
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

os.makedirs("results/visualizations", exist_ok=True)
os.makedirs("models", exist_ok=True)

#Data Loading and Exploration
def load_and_explore_data(file_path):
    """Load dataset and print basic statistics."""

    print("\nLoading dataset...")
    df = pd.read_csv(file_path)

    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    price_stats = df["price"].describe()
    print("\nPrice Summary:")
    print(f"Mean   : ${price_stats['mean']:,.2f}")
    print(f"Median : ${price_stats['50%']:,.2f}")
    print(f"Min    : ${price_stats['min']:,.2f}")
    print(f"Max    : ${price_stats['max']:,.2f}")

    return df

#Preprocessing
def preprocess_data(df):
    """Prepare features and target variable."""

    X = df.drop(["price", "car_ID", "CarName"], axis=1)
    y = df["price"]

    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    feature_names = (
        list(numerical_cols)
        + list(preprocessor.named_transformers_["cat"].get_feature_names_out())
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    print("\nPreprocessing completed")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, feature_names, preprocessor

#Linear Regression
def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Linear Regression."""

    start = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100,
        "train_time": train_time,
        "pred_time": 0.001,
    }

    print("\nLinear Regression Results")
    print(f"R²   : {metrics['R2']:.4f}")
    print(f"MAE  : ${metrics['MAE']:,.2f}")
    print(f"RMSE : ${metrics['RMSE']:,.2f}")

    return model, metrics, y_pred

#KNN Regression
def train_knn_regression(X_train, y_train, X_test, y_test, feature_names):
    """Train and evaluate KNN Regression with GridSearch."""

    param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}

    start = time.time()
    grid = GridSearchCV(
        KNeighborsRegressor(),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    train_time = time.time() - start

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "MAPE": mean_absolute_percentage_error(y_test, y_pred) * 100,
        "train_time": train_time,
        "pred_time": 0.006,
    }

    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=42
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names[: X_train.shape[1]],
            "importance": perm.importances_mean,
        }
    ).sort_values("importance", ascending=False)

    print("\nKNN Regression Results")
    print(f"Best k : {grid.best_params_['n_neighbors']}")
    print(f"R²     : {metrics['R2']:.4f}")
    print(f"MAE    : ${metrics['MAE']:,.2f}")

    return best_model, metrics, y_pred, importance_df, grid.best_params_

#Model Comparison
def compare_models(lr_metrics, knn_metrics):
    """Compare model performance."""

    comparison = pd.DataFrame(
        {
            "Metric": ["R2", "MAE", "RMSE", "MAPE", "Train Time"],
            "Linear Regression": [
                lr_metrics["R2"],
                lr_metrics["MAE"],
                lr_metrics["RMSE"],
                lr_metrics["MAPE"],
                lr_metrics["train_time"],
            ],
            "KNN Regression": [
                knn_metrics["R2"],
                knn_metrics["MAE"],
                knn_metrics["RMSE"],
                knn_metrics["MAPE"],
                knn_metrics["train_time"],
            ],
        }
    )

    print("\nModel Comparison")
    print(comparison.to_string(index=False))

    return comparison

#Visualisations
def generate_visualisations(y_test, lr_pred, knn_pred, feature_importance, knn_params):
    """Generate and save charts."""

    #Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, lr_pred, alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "r--")
    axes[0].set_title("Linear Regression: Actual vs Predicted")
    axes[0].set_xlabel("Actual Price")
    axes[0].set_ylabel("Predicted Price")

    axes[1].scatter(y_test, knn_pred, alpha=0.6)
    axes[1].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "r--")
    axes[1].set_title(f"KNN Regression (k={knn_params['n_neighbors']})")
    axes[1].set_xlabel("Actual Price")
    axes[1].set_ylabel("Predicted Price")

    plt.tight_layout()
    plt.savefig("results/visualizations/actual_vs_predicted.png", dpi=300)
    plt.show()

    #Residual plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(lr_pred, y_test - lr_pred, alpha=0.6)
    axes[0].axhline(0, linestyle="--")
    axes[0].set_title("Linear Regression Residuals")

    axes[1].scatter(knn_pred, y_test - knn_pred, alpha=0.6)
    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("KNN Regression Residuals")

    plt.tight_layout()
    plt.savefig("results/visualizations/residuals.png", dpi=300)
    plt.show()

    #Feature importance
    top_features = feature_importance.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importances (KNN)")
    plt.xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("results/visualizations/feature_importance.png", dpi=300)
    plt.show()

#Main Execution
if __name__ == "__main__":
    try:
        df = load_and_explore_data(
            "/Users/fabiolazeth/AI CLASS/CarPrice_Assignment.csv"
        )

        X_train, X_test, y_train, y_test, features, preprocessor = preprocess_data(df)

        lr_model, lr_metrics, lr_pred = train_linear_regression(
            X_train, y_train, X_test, y_test
        )

        knn_model, knn_metrics, knn_pred, feature_importance, knn_params = (
            train_knn_regression(X_train, y_train, X_test, y_test, features)
        )

        compare_models(lr_metrics, knn_metrics)

        generate_visualisations(
            y_test, lr_pred, knn_pred, feature_importance, knn_params
        )

        joblib.dump(lr_model, "models/linear_regression_model.pkl")
        joblib.dump(knn_model, "models/knn_regression_model.pkl")
        joblib.dump(preprocessor, "models/preprocessor.pkl")

        feature_importance.to_csv(
            "results/feature_importance.csv", index=False
        )

        print("\nModels, charts, and results saved successfully.")

    except Exception as e:
        print("Error:", e)