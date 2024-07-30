from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from typing import List
from models import Customer, PredictionInput, PredictionOutput, PredictionResult
from crud import get_all_customers, insert_customer, retrain_model, insert_prediction
from database import get_database
import pandas as pd
import joblib
import tensorflow as tf

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model('saved_models/regularized_model.h5')

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
        new_model = retrain_model(customers_collection, model)
        joblib.dump(new_model, 'model.pkl')
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {e}")

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput, db = Depends(get_database)):
    try:
        input_df = pd.DataFrame([input_data.dict()])
        prediction = model.predict(input_df.drop(columns=['CustomerId']))
        predictions_collection = db['predictions']
        prediction_result = PredictionResult(
            CustomerId=input_data.CustomerId,
            Prediction=bool(prediction[0])
        )
        insert_prediction(predictions_collection, prediction_result.dict(by_alias=True))
        return PredictionOutput(CustomerId=input_data.CustomerId, Prediction=bool(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
