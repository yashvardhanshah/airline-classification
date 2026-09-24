import pandas as pd

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
]


def preprocess(df):
    df = df.copy()

    # cleaning
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])

    # encoding
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["Customer Type"] = (df["Customer Type"] == "Loyal Customer").astype(int)
    df["Type of Travel"] = (df["Type of Travel"] == "Business travel").astype(int)
    df["Class"] = df["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2})

    # engineered features
    df["avg_rating"] = df[RATING_COLS].mean(axis=1)
    df["zero_ratings"] = (df[RATING_COLS] == 0).sum(axis=1)
    df["is_delayed"] = (df["Departure Delay in Minutes"] > 15).astype(int)
    df["age_group"] = pd.cut(df["Age"], [0, 20, 30, 40, 50, 60, 120], labels=False)
    return df