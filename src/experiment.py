import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import os
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Создание папок, если они не существуют
os.makedirs('metrics', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Загрузка данных
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    df.to_csv('data/iris.csv', index=False)
    return df

def split_data(df):
    X = df.drop('Species', axis=1)
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, max_iter=200):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, max_iter):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {"accuracy": accuracy}
    
    # Сохраняем метрики в файл с уникальным именем
    metrics_filename = f'metrics/metrics_{max_iter}.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Accuracy: {accuracy}")
    return cm

def plot_confusion_matrix(cm, max_iter):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (max_iter={max_iter})')
    
    # Сохраняем график с уникальным именем
    plot_filename = f'plots/confusion_matrix_{max_iter}.png'
    plt.savefig(plot_filename)
    plt.close()

def run_experiment(max_iter=200):
    df = load_data()
    
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train, max_iter)
    
    cm = evaluate_model(model, X_test, y_test, max_iter)
    
    plot_confusion_matrix(cm, max_iter)
    
    # Сохраняем модель в файл с уникальным именем
    model_filename = f'models/iris_model_{max_iter}.joblib'
    dump(model, model_filename)

if __name__ == '__main__':
    # Парсинг аргумента max_iter
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations for LogisticRegression')
    args = parser.parse_args()

    # Запуск эксперимента с переданным значением max_iter
    run_experiment(max_iter=args.max_iter)
