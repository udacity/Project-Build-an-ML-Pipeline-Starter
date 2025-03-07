import wandb

# Initialize a new W&B run
wandb.init(project="your_project_name", name="upload_sample_data")

# Create an artifact and add the CSV file
artifact = wandb.Artifact("sample_data", type="dataset")
artifact.add_file("get_data/sample.csv")

# Log the artifact to W&B
wandb.log_artifact(artifact)

print("✅ sample.csv successfully uploaded to Weights & Biases!")


import wandb
import pandas as pd

# Start W&B run
run = wandb.init(project="nyc_airbnb", save_code=True)

# Use full artifact path (replace 'your_username' with your actual W&B username)
artifact_path = "your_username/nyc_airbnb/sample_data:latest"
artifact = wandb.use_artifact(artifact_path)

# Download the artifact
local_path = artifact.file()
df = pd.read_csv(local_path)

print("✅ sample.csv successfully downloaded!")


import wandb

api = wandb.Api()
artifacts = api.artifact_type("dataset", "your_username/nyc_airbnb").collections()
for artifact in artifacts:
    print(artifact.name)
