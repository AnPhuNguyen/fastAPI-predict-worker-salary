import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def load_model(pickle_path='module_pickle.pkl'):
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_salary(features, model=None):
    """
    Predict salary given features using the loaded model and return prediction and model accuracy.

    Parameters:
    - features: list or array-like of length 3, representing the input features in the order used in training.
    - model: pre-loaded sklearn model. If None, loads from default pickle path.

    Returns:
    - dict with keys:
      - "result": predicted salary (float)
      - "accuracy": model accuracy (R2 score as float)
    """
    if model is None:
        model = load_model()

    # Load training data to get feature columns and test set for accuracy calculation
    df = pd.read_csv('salaries_2.csv')
    x = pd.DataFrame(df.iloc[:, 0:3])
    y = pd.Series(df.iloc[:, 3])

    # Split data to get test set for accuracy calculation
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Predict on input features
    input_df = pd.DataFrame([features], columns=x.columns)
    prediction = model.predict(input_df)[0]

    # Predict on test set to calculate accuracy
    y_pred_test = model.predict(x_test)
    accuracy = r2_score(y_test, y_pred_test)

    return {"result": round(float(prediction),2), "accuracy": round(accuracy*100,2)}

if __name__ == "__main__":
    print(predict_salary([5,3,0]))