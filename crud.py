from pymongo.collection import Collection
from bson import ObjectId
import pandas as pd
from typing import List
from models import Customer, PredictionResult

def get_all_customers(customers_collection: Collection) -> List[Customer]:
    customers = customers_collection.find()
    return [Customer(**customer) for customer in customers]

def insert_customer(customers_collection: Collection, customer_data: dict) -> Customer:
    result = customers_collection.insert_one(customer_data)
    customer_data['_id'] = result.inserted_id
    return Customer(**customer_data)

def retrain_model(customers_collection: Collection, model):
    data = pd.DataFrame(list(customers_collection.find()))
    if not data.empty:
        X = data.drop(columns=['_id', 'CustomerId', 'Surname', 'Churned'])
        y = data['Churned']
        model.fit(X, y)
        return model
    else:
        raise ValueError("No data available for training")

def insert_prediction(predictions_collection: Collection, prediction_data: dict) -> PredictionResult:
    result = predictions_collection.insert_one(prediction_data)
    prediction_data['_id'] = result.inserted_id
    return PredictionResult(**prediction_data)
