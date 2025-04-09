
import os
import mlflow

def promote_model():
        
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

        client = mlflow.MlflowClient()

        model_name = "Rakesh_kgpian"
        # Get the latest version
        latest_version_staging = client.get_latest_versions(model_name, stages=["Stages"])[0].version

        # Archieve the current production model
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for version in prod_versions:
              client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage='Archived'
              )

        # Promote the new model to production
        client.transition_model_version_stage(
              name=model_name,
              version=latest_version_staging,
              stage="Production"
        )
        print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
      promote_model()


