from pydantic import BaseModel, Field, condecimal
from typing import Optional
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class Customer(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    CustomerId: int
    Surname: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: condecimal(max_digits=10, decimal_places=2)
    NumOfProducts: int
    HasCrCard: bool
    IsActiveMember: bool
    EstimatedSalary: condecimal(max_digits=10, decimal_places=2)
    Churned: bool

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class PredictionInput(BaseModel):
    CustomerId: int
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: condecimal(max_digits=10, decimal_places=2)
    NumOfProducts: int
    HasCrCard: bool
    IsActiveMember: bool
    EstimatedSalary: condecimal(max_digits=10, decimal_places=2)

class PredictionOutput(BaseModel):
    CustomerId: int
    Prediction: bool

class PredictionResult(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    CustomerId: int
    Prediction: bool
    time_stamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
