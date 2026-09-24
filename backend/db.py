import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)


def save_prediction(data, prediction, probability, model_version):
    row = {**data, "model_version": model_version,
           "prediction": prediction, "probability_satisfied": probability}
    columns = ", ".join(row)
    values = ", ".join(f":{name}" for name in row)
    sql = text(f"INSERT INTO predictions ({columns}) VALUES ({values}) RETURNING id")
    with engine.begin() as conn:
        return conn.execute(sql, row).scalar_one()