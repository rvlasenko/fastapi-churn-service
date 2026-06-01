from pydantic import BaseModel


class TrainResponse(BaseModel):
    accuracy: float
    f1: float
    train_size: int
    test_size: int
