import yaml
from datetime import datetime
import subprocess

def update_timestamp():
    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Update only the timestamp inside model_trainer
    if "model_trainer" in params:
        params["model_trainer"]["timestamp"] = timestamp
    else:
        params["model_trainer"] = {"timestamp": timestamp}

    with open("params.yaml", "w") as f:
        yaml.dump(params, f)

    print(f"[INFO] Updated timestamp: {timestamp}")
    return timestamp

def run_pipeline():
    try:
        update_timestamp()
        subprocess.run(["dvc", "repro"], check=True)
        print("[INFO] Pipeline run completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] DVC pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
