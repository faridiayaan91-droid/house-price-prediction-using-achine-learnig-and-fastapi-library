import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="California House Price Prediction API",
    version="1.0.0"
)

# Load model and feature names
model = joblib.load("house_model.joblib")
features = joblib.load("house_feature.joblib")


class HouseFeatures(BaseModel):
    MedInc: float = Field(
        gt=0,
        description="Median income of the neighbourhood"
    )

    HouseAge: float = Field(
        gt=0,
        description="Average age of houses in the block"
    )

    AveRooms: float = Field(
        gt=0,
        description="Average number of rooms per house"
    )

    AveBedrms: float = Field(
        gt=0,
        description="Average number of bedrooms per house"
    )

    Population: float = Field(
        gt=0,
        description="Population in the block"
    )

    AveOccup: float = Field(
        gt=0,
        description="Average occupants per household"
    )

    Latitude: float = Field(
        description="Latitude of the location"
    )

    Longitude: float = Field(
        description="Longitude of the location"
    )


@app.get("/")
def home():
    return {
        "message": "California House Price Prediction API",
        "status": "Running",
        "endpoint": "Send POST request to /predict"
    }


@app.get("/health")
def health():
    return {
        "status": "Running",
        "model": "RandomForestRegressor",
        "features": features,
        "average_error": "$39,000"
    }


@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        # Create DataFrame
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        # Arrange columns in correct order
        input_data = input_data[features]

        # Predict
        predicted = model.predict(input_data)[0]

        # Convert to dollars
        price_usd = predicted * 100000

        return {
            "success": True,
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"{predicted:.2f} hundred thousand",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}",
            "input": house.model_dump()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )