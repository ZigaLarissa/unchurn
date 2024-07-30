import pandas as pd
import tensorflow as tf
from pymongo.collection import Collection
from typing import List
from models import Customer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

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
    label_encoder_geography = LabelEncoder()
    label_encoder_gender = LabelEncoder()
    df['Geography'] = label_encoder_geography.fit_transform(df['Geography'])
    df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])
    
    # Split the data into features and target
    X = df.drop('Churned', axis=1)
    y = df['Churned']
    
    # Normalize the data
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Save preprocessors
    joblib.dump(label_encoder_geography, 'saved_models/label_encoder_geography.pkl')
    joblib.dump(label_encoder_gender, 'saved_models/label_encoder_gender.pkl')
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    
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

def evaluate_model(customers_collection):
    # Load data from the collection
    data = list(customers_collection.find())
    df = pd.DataFrame(data)
    
    # Drop unnecessary columns
    df = df.drop(['CustomerId', 'Surname', '_id'], axis=1)
    
    # Load preprocessors
    label_encoder_geography = joblib.load('saved_models/label_encoder_geography.pkl')
    label_encoder_gender = joblib.load('saved_models/label_encoder_gender.pkl')
    scaler = joblib.load('saved_models/scaler.pkl')
    
    # Encode categorical variables
    df['Geography'] = label_encoder_geography.transform(df['Geography'])
    df['Gender'] = label_encoder_gender.transform(df['Gender'])
    
    # Split the data into features and target
    X = df.drop('Churned', axis=1)
    y = df['Churned']
    
    # Normalize the data
    X_normalized = scaler.transform(X)
    
    # Load the model
    model = tf.keras.models.load_model('saved_models/regularized_model.h5')
    
    # Make predictions
    y_pred = model.predict(X_normalized)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    f1 = f1_score(y, y_pred_binary)
    auc_roc = roc_auc_score(y, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc
    }
