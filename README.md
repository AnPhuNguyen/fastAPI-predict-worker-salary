requirement:fastapi, uvicorn, scikit-learn, pandas, pydantic

# Salary Prediction FastAPI App

This project is a simple FastAPI application that predicts salary based on three input features:
- Years of experience
- Number of required skills met
- Whether the worker is in a union (0 or 1)

## Project Structure

- `predict_salary.py`: Contains the machine learning model loading and prediction logic. It loads a pre-trained model from a pickle file and uses it to predict salary and calculate model accuracy.
- `main.py`: The FastAPI application exposing a REST API with:
  - A POST endpoint `/predict_salary` that accepts the three input features and returns the predicted salary and model accuracy.
  - Serves a frontend web page for user input.
- `static/index.html`: A simple frontend web page that allows users to input the features and get salary predictions from the API.
- `module_pickle.pkl`: The pre-trained machine learning model used for prediction.
- `salaries_2.csv`: Dataset used for training and accuracy calculation.

## How It Works

1. The user accesses the frontend web page served at the root URL.
2. The user inputs the years of experience, number of skills met, and union status.
3. The frontend sends a POST request to the `/predict_salary` endpoint with the input data.
4. The FastAPI backend calls the `predict_salary` function from `predict_salary.py` to get the predicted salary and model accuracy.
5. The prediction result is returned to the frontend and displayed to the user.

## Running the App

1. Install dependencies:
   ```
   pip install fastapi uvicorn scikit-learn pandas pydantic
   ```
2. Run the FastAPI app:
   ```
   uvicorn main:app --reload
   ```
3. Open your browser and go to `http://localhost:8000` to access the frontend.

## Notes

- The model is a linear regression trained on the dataset `salaries_2.csv`.
- The prediction accuracy is reported as the R2 score percentage.
- The frontend is a simple HTML page using JavaScript fetch API to communicate with the backend.

## Ignored Files

- `Python_ML_Daily_2.ipynb` and `salaries_2_ttspl.py` contain exploratory and training code for the model and are not part of the deployed app.
