from pydantic import BaseModel


class Survived(BaseModel):
    name: str
    api_version: str
    model_version: str
