# Customer Churn Prediction with MLP Classifier

This project focuses on predicting customer churn using a Multi-Layer Perceptron (MLP) classifier. Two models are trained: one unregularized and another with L2 regularization, dropout, and early stopping.

## Project Structure

- `data/`: Contains the `Customer_Churn` dataset.
- `requirements.txt`: Lists the necessary Python modules for the project.
- `notebook.ipynb`: notebook that contains the process of training the MLP classifiers.

## Setup

### Virtual Environment

Ensure you have Python 3.11 installed. Create and activate a virtual environment:

```sh
python3.11 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies

Install the required packages from requirements.txt:

```
pip install -r requirements.txt
```

## Usage

The notebook ```notebook.ipynb``` includes the necessary code to load the dataset, preprocess the data, and train both the unregularized and regularized MLP classifiers.

### Notes
The data folder should contain the Customer_Churn dataset in a compatible format (e.g., CSV).

**PS:** This project is built on a virtual environment supported in Python 3.11.
