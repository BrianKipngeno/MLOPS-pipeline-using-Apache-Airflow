# MLOPS-pipeline-using-Apache-Airflow

## Overview
This project demonstrates how to build an MLOps pipeline using Apache Airflow to automate data preprocessing, model training, and deployment tasks. The dataset contains app usage behavior with key features such as:

- **Date** (usage day)
- **App** (e.g., Instagram, WhatsApp)
- **Usage** (minutes spent)
- **Notifications** (alerts received)
- **Times Opened** (app launches)

The goal is to automate the preprocessing of screentime data and use machine learning to predict app usage.

## Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, feature engineering, and normalization.
- **Model Training**: Using a Random Forest Regressor to predict app usage.
- **Automated Workflow**: Utilizing Apache Airflow to schedule and run daily preprocessing tasks.

## Installation
To set up the project, install the required dependencies:
```sh
pip install apache-airflow pandas scikit-learn
```

## Dataset
The dataset  `screentime_analysis.csv` is in the project directory.

## Data Preprocessing
The preprocessing script performs the following steps:
- Loads the dataset
- Converts the **Date** column to datetime format and extracts temporal features
- Encodes the categorical **App** column using one-hot encoding
- Scales numerical columns using `MinMaxScaler`
- Performs feature engineering by creating lagged and interaction features
- Saves the preprocessed data to a new CSV file

### Example Code:
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('screentime_analysis.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data = pd.get_dummies(data, columns=['App'], drop_first=True)
scaler = MinMaxScaler()
data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])
data.to_csv('preprocessed_screentime_analysis.csv', index=False)
```

## Model Training
The Random Forest model is trained using the preprocessed data:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

X = data.drop(columns=['Usage (minutes)', 'Date'])
y = data['Usage (minutes)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')
```

## Automating with Apache Airflow
Apache Airflow enables task automation using Directed Acyclic Graphs (DAGs). This project uses a DAG to preprocess data daily.

### Install Apache Airflow
```sh
pip install apache-airflow
```

### Define the DAG:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def preprocess_data():
    data = pd.read_csv('screentime_analysis.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    data = pd.get_dummies(data, columns=['App'], drop_first=True)
    scaler = MinMaxScaler()
    data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])
    data.to_csv('preprocessed_screentime_analysis.csv', index=False)

dag = DAG(
    dag_id='data_preprocessing',
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

preprocess_task = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess_data,
    dag=dag,
)
```

## Running the Pipeline
### Initialize Airflow Database
```sh
airflow db init
```

### Start Airflow Webserver
```sh
airflow webserver --port 8080
```

### Start Airflow Scheduler
```sh
airflow scheduler
```

### Access Airflow UI
Navigate to [http://localhost:8080](http://localhost:8080) in your browser, enable the DAG, and trigger it manually.

## Summary
This project showcases an **MLOps pipeline using Apache Airflow** to automate data preprocessing, train a **Random Forest model**, and deploy a **scheduled workflow** for continuous machine learning model updates. Apache Airflow ensures scalability, efficiency, and reproducibility in machine learning workflows.

