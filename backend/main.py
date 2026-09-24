from pathlib import Path
from typing import Annotated, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.preprocessing import preprocess

ARTIFACTS = Path(__file__).resolve().parent.parent / "ml" / "artifacts"
model = joblib.load(ARTIFACTS / "random_forest.joblib")
feature_columns = joblib.load(ARTIFACTS / "feature_columns.joblib")

app = FastAPI(title="Airline Satisfaction API")

Rating = Annotated[int, Field(ge=0, le=5)]


class Passenger(BaseModel):
    gender: Literal["Male", "Female"]
    customer_type: Literal["Loyal Customer", "disloyal Customer"]
    age: int = Field(ge=1, le=120)
    type_of_travel: Literal["Business travel", "Personal Travel"]
    travel_class: Literal["Eco", "Eco Plus", "Business"]
    flight_distance: int = Field(ge=0)
    inflight_wifi: Rating
    departure_arrival_time: Rating
    online_booking: Rating
    gate_location: Rating
    food_and_drink: Rating
    online_boarding: Rating
    seat_comfort: Rating
    inflight_entertainment: Rating
    onboard_service: Rating
    leg_room: Rating
    baggage_handling: Rating
    checkin_service: Rating
    inflight_service: Rating
    cleanliness: Rating
    departure_delay: int = Field(ge=0)
    arrival_delay: Optional[int] = Field(default=None, ge=0)


# form field name -> column name used in the training data
COLUMN_NAMES = {
    "gender": "Gender",
    "customer_type": "Customer Type",
    "age": "Age",
    "type_of_travel": "Type of Travel",
    "travel_class": "Class",
    "flight_distance": "Flight Distance",
    "inflight_wifi": "Inflight wifi service",
    "departure_arrival_time": "Departure/Arrival time convenient",
    "online_booking": "Ease of Online booking",
    "gate_location": "Gate location",
    "food_and_drink": "Food and drink",
    "online_boarding": "Online boarding",
    "seat_comfort": "Seat comfort",
    "inflight_entertainment": "Inflight entertainment",
    "onboard_service": "On-board service",
    "leg_room": "Leg room service",
    "baggage_handling": "Baggage handling",
    "checkin_service": "Checkin service",
    "inflight_service": "Inflight service",
    "cleanliness": "Cleanliness",
    "departure_delay": "Departure Delay in Minutes",
    "arrival_delay": "Arrival Delay in Minutes",
}


@app.post("/predict")
def predict(passenger: Passenger):
    raw = pd.DataFrame([{COLUMN_NAMES[k]: v for k, v in passenger.model_dump().items()}])
    raw["Arrival Delay in Minutes"] = raw["Arrival Delay in Minutes"].astype(float)

    features = preprocess(raw)[feature_columns]
    prob = float(model.predict_proba(features)[0, 1])

    return {
        "prediction": "satisfied" if prob >= 0.5 else "neutral or dissatisfied",
        "probability_satisfied": round(prob, 4),
    }