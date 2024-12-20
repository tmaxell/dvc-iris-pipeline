import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import os

os.makedirs('metrics', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

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

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    

    metrics = {"accuracy": accuracy}
    with open('metrics/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Accuracy: {accuracy}")
    return cm

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.show()

def main():
    df = load_data()
    
    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    
    cm = evaluate_model(model, X_test, y_test)
    
    plot_confusion_matrix(cm)
    
    from joblib import dump
    dump(model, 'models/iris_model.joblib')

if __name__ == '__main__':
    main()
