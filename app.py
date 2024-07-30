from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from models import Customer, PredictionInput, PredictionOutput, PredictionResult
from crud import get_all_customers, insert_customer, retrain_model, insert_prediction
from database import get_database
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
        new_model.save('saved_models/regularized_model.h5')
        return {"message": "Model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retraining model: {e}")

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput, db = Depends(get_database)):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Encode categorical variables
        label_encoder_geography = LabelEncoder()
        label_encoder_gender = LabelEncoder()
        input_df['Geography'] = label_encoder_geography.fit_transform(input_df['Geography'])
        input_df['Gender'] = label_encoder_gender.fit_transform(input_df['Gender'])
        
        # Normalize the data
        scaler = StandardScaler()
        X = input_df.drop(columns=['CustomerId'])
        X_normalized = scaler.fit_transform(X)
        
        # Make prediction
        prediction = model.predict(X_normalized)
        prediction_bool = bool(prediction[0][0])
        
        # Map the prediction to a label
        prediction_label = "Churned" if prediction_bool else "Unchurned"
        
        # Store the prediction result
        predictions_collection = db['predictions']
        prediction_result = PredictionResult(
            CustomerId=input_data.CustomerId,
            Prediction=prediction_bool
        )
        insert_prediction(predictions_collection, prediction_result.dict(by_alias=True))
        
        # Return the prediction
        return PredictionOutput(
            CustomerId=input_data.CustomerId,
            Prediction=prediction_bool,
            PredictionLabel=prediction_label
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")
