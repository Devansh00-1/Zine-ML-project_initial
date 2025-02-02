import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def evaluate_model(y_test, y_pred, y_pred_probs=None, title="Confusion Matrix", text_file=None, csv_file=None):
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    precision = report['1']['precision'] if '1' in report else 0
    recall = report['1']['recall'] if '1' in report else 0
    f1 = report['1']['f1-score'] if '1' in report else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    if y_pred_probs is not None:
        # Add ROC AUC to the classification report
        try:
            roc_auc = roc_auc_score(y_test, y_pred_probs)
            print(f"ROC AUC: {roc_auc:.4f}")
        except ValueError:
            print("ROC AUC cannot be calculated. Ensure y_test and y_pred_probs are valid.")
    if text_file:
        with open(text_file, 'a') as f:
            f.write(f"{title}\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\nAccuracy Score:\n")
            f.write(f"{accuracy}\n")
            if y_pred_probs is not None:
                f.write(f"ROC AUC: {roc_auc:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(f"{cm}\n\n")

    if csv_file:
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([title])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Accuracy", accuracy])
            writer.writerow(["Precision", precision])
            writer.writerow(["Recall", recall])
            writer.writerow(["F1-score", f1])
            if y_pred_probs is not None:
                writer.writerow(["ROC AUC", roc_auc])
            writer.writerow([])
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        writer.writerow([f"{label} {metric}", value])
            writer.writerow([])

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    plt.show()

    if y_pred_probs is not None:
        plot_roc_curve(y_test, y_pred_probs, title="ROC Curve")

def plot_roc_curve(y_test, y_pred_probs, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def load_and_split_data(train_file, test_file):
    try:
        x_train = pd.read_csv(train_file).dropna()
        y_train = x_train.pop('label')
        x_test = pd.read_csv(test_file).dropna()
        y_test = x_test.pop('label')

        if 'id' in x_train.columns:
            x_train.drop(columns=['id'], inplace=True)
        if 'id' in x_test.columns:
            x_test.drop(columns=['id'], inplace=True)
        # Encode categorical features
        label_encoders = {}
        for column in x_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            x_train[column] = le.fit_transform(x_train[column])
            # Applying samye encoding on test data
            x_test[column] = le.transform(x_test[column])
            label_encoders[column] = le # Save the encoder for later use
        
        for column in x_test.select_dtypes(include=['object']).columns:
            if column in label_encoders:
                x_test[column] = x_test[column].map(lambda s: label_encoders[column].transform([s])[0] if s in label_encoders[column].classes_ else -1)
            else:
                x_test[column] = -1 # Default unknown categories to -1
            
        # Ensure both datasets have the same features
        x_train, x_test = x_train.align(x_test, join='inner', axis=1)
    except FileNotFoundError:
        print(f"Error: The file at {train_file} or {test_file} was not found.")
        return None, None, None, None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None, None, None, None
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
        return None, None, None, None

    return x_train, x_test, y_train, y_test

def neuralNetworkClassifier(train_file, test_file, text_file, csv_file):
    X_train, X_test, y_train, y_test = load_and_split_data(train_file, test_file)
    if X_train is None:
        return

    try:
        model = Sequential([
            Dense(32, input_dim=X_train.shape[1], activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    plt.figure(figsize=(10, 7))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Neural Network Training Loss')
    plt.legend()
    plt.show()

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Neural Network Accuracy: {accuracy}')
    
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_probs = model.predict(X_test)
    evaluate_model(y_test, y_pred, y_pred_probs, "Neural Network Confusion Matrix", text_file, csv_file)

def decisionTreeClassfier(train_file, test_file, text_file, csv_file):
    X_train, X_test, y_train, y_test = load_and_split_data(train_file, test_file)
    if X_train is None:
        return

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 5, 10]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = DecisionTreeClassifier(
        criterion=grid_search.best_params_['criterion'],
        max_depth=grid_search.best_params_['max_depth'], 
        min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
        min_samples_split=grid_search.best_params_['min_samples_split'],
        class_weight='balanced')
    
    try:
        best_model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    y_pred = best_model.predict(X_test)
    y_pred_probs = best_model.predict_proba(X_test)[:, 1]

    print(f"Best Parameters: {grid_search.best_params_}")
    
    evaluate_model(y_test, y_pred, y_pred_probs, "Decision Tree with Optimized Hyperparameters Confusion Matrix", text_file, csv_file)

def randomForestClassifier(train_file, test_file, text_file, csv_file):
    X_train, X_test, y_train, y_test = load_and_split_data(train_file, test_file)
    if X_train is None:
        return

    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    try:
        random_search = RandomizedSearchCV(model, param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1)
        random_search.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    staged_losses = []
    for n_estimators in range(1, 101):
        model.set_params(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        staged_losses.append(1 - model.score(X_test, y_test))
    
    plt.figure(figsize=(10, 7))
    plt.plot(range(1, 101), staged_losses, label='Random Forest Loss')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Loss')
    plt.title('Random Forest Staged Loss')
    plt.legend()
    plt.show()
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_probs = best_model.predict_proba(X_test)[:, 1]
    
    print(f"Best Parameters: {random_search.best_params_}")
    print(f'Random Forest Accuracy: {best_model.score(X_test, y_test)}')
    evaluate_model(y_test, y_pred, y_pred_probs, "Random Forest Confusion Matrix", text_file, csv_file)

def main(train_file, test_file):
    
    text_file = "output.txt"
    csv_file = "metrics.csv"

    open(text_file, 'w').close()
    open(csv_file, 'w').close()

    decisionTreeClassfier(train_file, test_file, text_file, csv_file)
    randomForestClassifier(train_file, test_file, text_file, csv_file)
    neuralNetworkClassifier(train_file, test_file, text_file, csv_file)

if __name__ == "__main__":
    train_file = r"C:\Users\radhika\OneDrive\Desktop\Zine projects\ZINE-ML-PROJECT_INITIAL\mytrainingdata.csv"
    test_file = r"C:\Users\radhika\OneDrive\Desktop\Zine projects\ZINE-ML-PROJECT_INITIAL\archive\UNSW_NB15_testing-set.csv"
    
    main(train_file, test_file)
