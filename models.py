from pydantic import BaseModel, Field, condecimal
from bson import ObjectId
from datetime import datetime
from decimal import Decimal

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema, field):
        field_schema.update(type="string")
        return field_schema

class Customer(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    CustomerId: int
    
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: Decimal
    NumOfProducts: int
    HasCrCard: bool
    IsActiveMember: bool
    EstimatedSalary: Decimal
    Churned: bool

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class PredictionInput(BaseModel):
    CustomerId: int
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: Decimal
    NumOfProducts: int
    HasCrCard: bool
    IsActiveMember: bool
    EstimatedSalary: Decimal

class PredictionOutput(BaseModel):
    CustomerId: int
    Prediction: bool
    PredictionLabel: str

    class Config:
        json_encoders = {ObjectId: str}
