from pymongo.collection import Collection
from bson import ObjectId
import pandas as pd
from typing import List
from models import Customer, PredictionResult
import tensorflow as tf


def get_all_customers(customers_collection: Collection) -> List[Customer]:
    customers = customers_collection.find()
    return [Customer(**customer) for customer in customers]

def insert_customer(customers_collection: Collection, customer_data: dict) -> Customer:
    result = customers_collection.insert_one(customer_data)
    customer_data['_id'] = result.inserted_id
    return Customer(**customer_data)


def retrain_model(collection: Collection, model: tf.keras.Model):
    data = get_all_customers(collection)
    df = pd.DataFrame(data)

    # Ensure the dataframe has the expected columns
    expected_columns = ["CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Churned"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing columns in the dataframe: {missing_columns}")

    # Drop the unnecessary columns
    df.drop(columns=["CustomerId", "Surname", "Churned"], inplace=True, errors='ignore')
    X = df.drop(columns=["Churned"])
    y = df["Churned"]

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32)

    return model


def insert_prediction(predictions_collection: Collection, prediction_data: dict) -> PredictionResult:
    result = predictions_collection.insert_one(prediction_data)
    prediction_data['_id'] = result.inserted_id
    return PredictionResult(**prediction_data)
