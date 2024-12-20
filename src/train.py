import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import os

# Загрузка данных
data = pd.read_csv("data/iris.csv", header=None)
data.columns = ["Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]

# Исключение столбца "Id" из данных
X = data.drop(["Id", "Species"], axis=1)
y = data["Species"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Оценка качества
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Сохранение модели и метрик
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    import pickle
    pickle.dump(model, f)

metrics = {"accuracy": accuracy}
os.makedirs("metrics", exist_ok=True)
with open("metrics/results.json", "w") as f:
    json.dump(metrics, f)

print(f"Accuracy: {accuracy}")