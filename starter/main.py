# Put the code for your API here.

from fastapi import FastAPI, Request
import os, joblib
import pandas as pd
from pydantic import BaseModel, Field, validator
from typing import List
# from starter.model import inference

app = FastAPI()

CWD = os.getcwd()

@app.get("/")
def read_root():
    return {"msg": "Hello World"}

# class Census(BaseModel):
#  workclass: str
#  education: str
#  marital_status: str = Field(alias="marital-status")
#  occupation: str
#  relationship:str
#  race:str
#  sex:str
#  native_country:str = Field(alias="native-country")


class Census(BaseModel):
 workclass: str
 education: str
 marital_status: str = Field(alias='marital-status')
 occupation: str
 relationship:str
 race:str
 sex:str
 native_country:str = Field(alias='native-country')

model_filename = os.path.join(CWD, 'model', 'rf_model.pkl')
encoder_filename = os.path.join(CWD, 'model', 'encoder.pkl')
model = joblib.load(model_filename)
encoder=joblib.load(encoder_filename)

@app.post('/inference')
async def inference(census: Census):
  
 # Converting JSON to Pandas DataFrame
 input_df = pd.DataFrame([census.dict()])
 
 # Getting the prediction from the RandomForest model
 input_x = encoder.transform(input_df)
 pred = model.predict(input_x)[0]
 
 return int(pred)