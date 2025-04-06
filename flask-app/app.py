
from flask import Flask, render_template, request, jsonify
import mlflow.pyfunc
import os
from mlflow.tracking import MlflowClient
from preprocessing_utility import normalize_text
import pickle

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

app = Flask(__name__)

# 🔹 Set MLflow tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/rakeshkumar93694/mlops-mini-project.mlflow")

# 🔹 Authenticate with DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "rakeshkumar93694"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "7f7873035be490d229f82762b0b08ab1296f6570"

# 🔹 Fetch the latest model version dynamically
def get_latest_model_version(model_name):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        return None  # No versions available

    latest_version = max(versions, key=lambda v: int(v.version)).version  # Get highest version
    return latest_version

# 🔹 Load model from DagsHub MLflow Model Registry
model_name = "Rakesh_kgpian"
model_version = get_latest_model_version(model_name)

if model_version is None:
    raise ValueError(f"No version found for model: {model_name}")

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

@app.route('/')
def home():
    return render_template('index.html',result=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # 🔹 Clean text using preprocessing utility
    text = normalize_text(text)
    
    # tfidf
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)

    # show
    return render_template('index.html', result=result[0])

if __name__ == '__main__':
    app.run(debug=True, port=5001)
