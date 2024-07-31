# Customer Churn Prediction with MLP Classifier and FastAPI

This project focuses on predicting customer churn using a Multi-Layer Perceptron (MLP) classifier, with a FastAPI backend for data management, model retraining, and predictions.

## Project Structure

- `data/`: Contains the `Customer_Churn` dataset.
- `requirements.txt`: Lists the necessary Python modules for the project.
- `notebook.ipynb`: Jupyter notebook that contains the process of training the MLP classifiers.
- `app.py`: Contains the FastAPI routes and application logic.
- `crud.py`: Contains CRUD operations for database interactions.
- `models.py`: Defines MongoDB database models.
- `database.py`: Handles MongoDB database connection and operations.
- `saved_models/`: Directory for storing trained models.

## Setup

### Virtual Environment

Ensure you have Python 3.11 installed. Create and activate a virtual environment:

```sh
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### Install Dependencies

Install the required packages from requirements.txt:

```sh
pip install -r requirements.txt
```

## Usage

### Running the FastAPI Application

To run the project locally, ensure you are in the `unchurn` repository and run:

```sh
uvicorn app:app --reload
```

This will start the FastAPI server with hot reloading enabled.

### API Endpoints

The following endpoints are available:

- `POST /upload-data`: Upload a CSV file containing customer data.
- `POST /retrain`: Retrain the model using the latest data.
- `POST /predict`: Make predictions using the trained model. Requires a JSON payload with customer data.
- `GET /evaluate-model`: Evaluate the current model's performance.

For detailed information on request/response formats, please refer to the API documentation.

### Deployed API

The API is deployed and accessible at: https://unchurn-model.onrender.com

You can interact with the API using the Swagger UI available at this URL.

## Notes

- The project uses MongoDB as its database backend.
- Ensure you have the necessary MongoDB connection details configured in your environment.
- The `data` folder should contain the Customer_Churn dataset in a compatible format (e.g., CSV) if you're working with local data.

**PS:** This project is built on a virtual environment supported in Python 3.11.
```
