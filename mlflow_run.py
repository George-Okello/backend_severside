import os
import sys
import subprocess


def run_mlflow():
    # Run the training scripts
    subprocess.run([sys.executable, "scripts/train_model.py"])
    # Run the evaluation scripts
    subprocess.run([sys.executable, "scripts/evaluate_model.py"])


if __name__ == "__main__":
    run_mlflow()
