import pickle

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
   "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float

@app.get("/*")
async def index():
    return

@app.post("/pred")
async def root(payload: PredictionInput):
    with open("model.pickle", "rb") as f:
        model = pickle.load(f)
        input = np.asarray(
            (
                payload.ph,
                payload.hardness,
                payload.solids,
                payload.chloramines,
                payload.sulfate,
                payload.conductivity,
                payload.organic_carbon,
                payload.trihalomethanes,
                payload.turbidity,
            )
        ).reshape(1, -1)
        result = model.predict(input)
        return {"prediction": result[0]}
