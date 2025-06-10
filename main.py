from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from predict_salary import predict_salary

from fastapi.responses import FileResponse
import os

app = FastAPI()

class SalaryInput(BaseModel):
    years_experience: float
    skills_met: int
    in_union: int  # 0 or 1

# Mount frontend static files at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/", response_class=FileResponse)
async def root():
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/predict_salary")
def predict_salary_endpoint(input: SalaryInput):
    features = [input.years_experience, input.skills_met, input.in_union]
    result = predict_salary(features)
    return result
