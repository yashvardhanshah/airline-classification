import numpy as np
import pandas as pd
from sqlalchemy import text

from backend.db import engine
from backend.main import COLUMN_NAMES, MODEL_VERSION, feature_columns, model
from ml.preprocessing import preprocess

N_ROWS = 5000
DAYS = 60
rng = np.random.default_rng(42)

test = pd.read_csv("data/test.csv").sample(N_ROWS, random_state=42).reset_index(drop=True)
actual = test["satisfaction"]
raw = test.drop(columns=["Unnamed: 0", "id", "satisfaction"])

prob = model.predict_proba(preprocess(raw)[feature_columns])[:, 1]

rows = raw.rename(columns={v: k for k, v in COLUMN_NAMES.items()})
rows["arrival_delay"] = rows["arrival_delay"].astype("Int64")
rows["prediction"] = np.where(prob >= 0.5, "satisfied", "neutral or dissatisfied")
rows["probability_satisfied"] = prob.round(4)
rows["actual_label"] = actual
rows["model_version"] = MODEL_VERSION
rows["source"] = "seed"
rows["created_at"] = pd.Timestamp.now(tz="UTC") - pd.to_timedelta(
    rng.uniform(0, DAYS * 24 * 3600, N_ROWS), unit="s"
)

with engine.begin() as conn:
    conn.execute(text("DELETE FROM predictions WHERE source = 'seed'"))
rows.to_sql("predictions", engine, if_exists="append", index=False, method="multi", chunksize=1000)
print(f"Inserted {len(rows):,} seed rows")