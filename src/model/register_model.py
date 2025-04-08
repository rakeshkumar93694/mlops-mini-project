# import json
# import logging
# import mlflow
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


# def load_model_info(file_path: str) -> dict:
#     """Load the model info from a JSON file."""
#     try:
#         with open(file_path, 'r') as file:
#             model_info = json.load(file)
#         logger.debug('Model info loaded from %s', file_path)
#         return model_info

#     except FileNotFoundError:
#         logger.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error whilde loading the model file: %s',e)


# def register_model(model_name: str, model_info: dict):
#     try:
#         model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

#         # Register the model
#         model_version = mlflow.register_model(model_uri, model_name)

#         # Transition the model to "Staging" stage
#         client = mlflow.tracking.MlflowClient()
#         client.transition_model_version_stage(
#             name=model_name,
#             version=model_version.version,
#             stage="Staging"
#         )

#         logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned')

#     except Exception as e:
#         logger.error('Error during model registration: %s', e)
        
# def main():
#     try:
#         model_info_path = 'reports/experiment_info.json'
#         model_info = load_model_info(model_info_path)

#         model_name = "Rakesh_kgpian"
#         register_model(model_name, model_info)
#     except Exception as e:
#         logger.error('Failed to complete the model registration process: %s',e)
#         print(f"Error: {e}")

# if __name__ == '__main__':
#     main()


import json
import logging
import mlflow
import os
import time
from dotenv import load_dotenv
from mlflow.exceptions import MlflowException

# Load environment variables
load_dotenv()

# Set tracking URI and authentication
mlflow.set_tracking_uri('https://dagshub.com/rakeshkumar93694/mlops-mini-project.mlflow')
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading the model file: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        model_uri = f"runs:/{run_id}/{model_path}"

        logger.info(f"Attempting to register model from URI: {model_uri}")

        client = mlflow.tracking.MlflowClient()

        # Validate that the run exists
        try:
            run = client.get_run(run_id)
            logger.debug(f"Run ID {run_id} exists: {run.info.status}")
        except MlflowException as e:
            logger.error(f"Run ID {run_id} not found: {e}")
            return

        # Delay to handle DAGsHub syncing lag
        time.sleep(5)

        # Register the model
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        logger.info(f"Model '{model_name}' version {model_version.version} registered.")

        # Wait for the registration to complete
        time.sleep(5)

        # Transition to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f"Model {model_name} version {model_version.version} transitioned to 'Staging'")

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "Rakesh_kgpian"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

