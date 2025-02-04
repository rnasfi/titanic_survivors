from typing import Any, List, Optional

from classification_model.processing.validation import TitanicDataInputSchema
from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]


class MultiplePassengersDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "pclass": 1,
                        "sex": "female",
                        "age": 80.0,
                        "sibsp": 0,
                        "parch": 0,
                        "fare": 211.3375,
                        "cabin": "B12",
                        "embarked": "C",
                    },
                    {
                        "pclass": 3,
                        "sex": "female",
                        "age": 80.0,
                        "sibsp": 0,
                        "parch": 0,
                        "fare": 1000.3375,
                        "cabin": "B12",
                        "title": "Ms.",
                    }
                ]
            }
        }
