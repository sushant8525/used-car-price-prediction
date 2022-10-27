# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model
from fastapi import Depends


# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("app")

# Create input/output pydantic models
input_model = create_model("app_input", **{'Location': 'Coimbatore', 'Year': 2018, 'Kilometers_Driven': 41467, 'Fuel_Type': 1, 'Transmission': 2, 'Owner_Type': 1, 'Mileage': '15.0', 'Engine': '2143', 'Power': '204 ', 'Seats': 5.0, 'Manufacturer': 'Mercedes-Benz'})
output_model = create_model("app_output", Price_prediction=37.98)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model = Depends()):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"Price_prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
