from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from models import Customer, PredictionInput, PredictionOutput
from crud import get_all_customers, insert_customer, retrain_model, insert_prediction, evaluate_model
from database import get_database
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...), db = Depends(get_database)):
    try:
        data = pd.read_csv(file.file)
        customers_collection = db['customers']
        for record in data.to_dict('records'):
            insert_customer(customers_collection, record)
        return {"message": "Data uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading data: {e}")

@app.post("/retrain")
async def retrain_model_endpoint(db = Depends(get_database)):
    try:
        customers_collection = db['customers']
        new_model = retrain_model(customers_collection, None)  # Pass None as we're not using an existing model
        new_model.save('saved_models/regularized_model.h5')
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {e}")

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput, db = Depends(get_database)):
    try:
        logger.info(f"Received input data: {input_data}")
        
        # Load model
        model = tf.keras.models.load_model('saved_models/regularized_model.h5')
        logger.info("Model loaded successfully")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        logger.info(f"Input DataFrame: {input_df}")
        
        # Load and apply preprocessors
        label_encoder_geography = joblib.load('saved_models/label_encoder_geography.pkl')
        label_encoder_gender = joblib.load('saved_models/label_encoder_gender.pkl')
        scaler = joblib.load('saved_models/scaler.pkl')
        
        input_df['Geography'] = label_encoder_geography.transform(input_df['Geography'])
        input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
        logger.info(f"Encoded DataFrame: {input_df}")
        
        X = input_df.drop(columns=['CustomerId'])
        X_normalized = scaler.transform(X)
        logger.info(f"Normalized data shape: {X_normalized.shape}")
        
        # Make prediction
        prediction = model.predict(X_normalized)
        prediction_bool = bool(prediction[0][0] > 0.5)
        logger.info(f"Raw prediction: {prediction}, Thresholded prediction: {prediction_bool}")
        
        # Map the prediction to a label
        prediction_label = "Churned" if prediction_bool else "Not Churned"
        
        # Store the prediction result
        predictions_collection = db['predictions']
        prediction_output = PredictionOutput(
            CustomerId=input_data.CustomerId,
            Prediction=prediction_bool,
            PredictionLabel=prediction_label
        )
        insert_prediction(predictions_collection, prediction_output.dict(by_alias=True))
        
        logger.info(f"Final prediction output: {prediction_output}")
        
        # Return the prediction
        return prediction_output
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")


@app.get("/evaluate-model")
async def evaluate_model_endpoint(db = Depends(get_database)):
    try:
        customers_collection = db['customers']
        evaluation_results = evaluate_model(customers_collection)
        return evaluation_results
    except Exception as e:
        logger.error(f"Error in evaluate model endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluating model: {e}")