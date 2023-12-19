import json

import pandas as pd
import mysql.connector
from typing import Dict

mysql_message_json_file: str = "mysql_message.json"
model_csv_file: str = "model_data.csv"

with open(mysql_message_json_file, "r") as json_file:
    mysql_data: Dict[str, str] = json.load(json_file)

connection = mysql.connector.connect(
    host=mysql_data["host"],
    user=mysql_data["user"],
    port=mysql_data["port"],
    password=mysql_data["password"],
    database=mysql_data["database"],
)

cursor = connection.cursor()
cursor.close()

model_datas: pd.DataFrame = pd.read_csv(model_csv_file)
print(model_datas["alpha"])
