
from flask import Flask, render_template, request, jsonify
import mlflow.pyfunc
import os
from mlflow.tracking import MlflowClient
# from flask_app.preprocessing_utility import normalize_text
import pickle
import pandas as pd
import string
from string import ascii_lowercase




def normalize_text(text):
    text = lower_case(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# Set up DagsHub credentials for MLflow tracking

dagshub_token = os.getenv("DAGSHUB_KGPIAN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_Kgpian environment variable is not set")

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "rakeshkumar93694"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

app = Flask(__name__)

       

# load model from model registry

def get_latest_model_version(model_name):
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

model_name = "Rakesh_kgpian"
model_version = get_latest_model_version(model_name)

model_uri = f'models:/{model_name}/{model_version}'
       
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('model/vectorizer.pkl','rb'))


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

    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    # prediction
    result = model.predict(features_df)

    # show
    return render_template('index.html', result=result[0])

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")

