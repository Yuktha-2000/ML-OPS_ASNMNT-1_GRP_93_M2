from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
from preprocessing import preprocess_data, load_data, get_column_definitions, split_data

# Set the MLflow experiment
mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment('ML_Model_Experiment')

def print_performance_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{model_name} Precision: {precision * 100:.2f}%")
    print(f"{model_name} Recall: {recall * 100:.2f}%")
    print(f"{model_name} F1-Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Log metrics in MLflow
    mlflow.log_metric("accuracy", accuracy * 100)
    mlflow.log_metric("precision", precision * 100)
    mlflow.log_metric("recall", recall * 100)
    mlflow.log_metric("f1_score", f1 * 100)

def logistic_regression_pipeline(x_train, y_train, x_test, y_test):
    with mlflow.start_run(run_name="Logistic_Regression_Trial"):
        mlflow.log_param("model_name", "Logistic Regression")
        
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)
        print_performance_metrics(y_test, y_pred, 'Logistic Regression')
        
        mlflow.sklearn.log_model(lr, "model")  # Log model to MLflow
        return lr  # Return the trained model

def ada_boost_pipeline(x_train, y_train, x_test, y_test):
    with mlflow.start_run(run_name="AdaBoost_Trial"):
        mlflow.log_param("model_name", "AdaBoost")
        
        ada = AdaBoostClassifier(n_estimators=200, random_state=42)
        ada.fit(x_train, y_train)
        y_pred = ada.predict(x_test)
        print_performance_metrics(y_test, y_pred, 'AdaBoost')
        
        mlflow.sklearn.log_model(ada, "model")  # Log model to MLflow
        return ada  # Return the trained model

def random_forest_pipeline(x_train, y_train, x_test, y_test):
    with mlflow.start_run(run_name="Random_Forest_Trial"):
        mlflow.log_param("model_name", "Random Forest")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        print_performance_metrics(y_test, y_pred, 'Random Forest')
        
        mlflow.sklearn.log_model(rf, "model")  # Log model to MLflow
        return rf  # Return the trained model

def save_best_model(model, filepath):
    """Save the best-performing model to a file."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

# Example of how to run the trials
def train_models_and_log_results(x_train, y_train, x_test, y_test):
    # Run logistic regression model
    logistic_model = logistic_regression_pipeline(x_train, y_train, x_test, y_test)

    # Run AdaBoost model
    ada_model = ada_boost_pipeline(x_train, y_train, x_test, y_test)

    # Run Random Forest model
    rf_model = random_forest_pipeline(x_train, y_train, x_test, y_test)


df = load_data("data/Transactions_Dataset.csv")
cat_cols, num_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order = get_column_definitions()
X, y = preprocess_data(df, cat_cols, scale_cols, ordinal_cols, reputation_order, transaction_freq_order)
# Train/test split
x_train, x_test, y_train, y_test = split_data(X, y)
train_models_and_log_results(x_train, y_train, x_test, y_test)
