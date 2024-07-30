import pandas as pd
import tensorflow as tf
from pymongo.collection import Collection
from typing import List
from models import Customer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def get_all_customers(customers_collection: Collection) -> List[Customer]:
    customers = customers_collection.find()
    return [Customer(**customer) for customer in customers]

def insert_customer(customers_collection: Collection, customer_data: dict):
    result = customers_collection.insert_one(customer_data)
    customer_data['_id'] = result.inserted_id
    return customer_data

def insert_prediction(collection: Collection, data: dict):
    collection.insert_one(data)

def retrain_model(customers_collection, model):
    # Load data from the collection
    data = list(customers_collection.find())
    df = pd.DataFrame(data)
    
    # Drop unnecessary columns
    df = df.drop(['CustomerId', 'Surname', '_id'], axis=1)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Geography'] = label_encoder.fit_transform(df['Geography'])
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    
    # Split the data into features and target
    X = df.drop('Churned', axis=1)
    y = df['Churned']
    
    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Define the model architecture
    def l2_dropout_model(regularizer=None):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    # L2 regularization
    regularizer = tf.keras.regularizers.l2(0.001)
    new_model = l2_dropout_model(regularizer=regularizer)
    
    # Introduce early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Train the model with early stopping
    new_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    return new_model


