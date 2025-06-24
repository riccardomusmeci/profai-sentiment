import argparse
import sys
import json
from profai.utils.config import load_config
from profai.inference.predictor import SentimentPredictor

from transformers import logging
logging.set_verbosity_error()

def main():
    parser = argparse.ArgumentParser(description="ProfAI Inference Script")
    parser.add_argument("--config", type=str, required=False, help="Path to YAML config file", default="config/predict.yaml")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    args = parser.parse_args()

    config = load_config(args.config)

    model_id = config["model"]["model_id"]
    num_labels = config["model"].get("num_labels", 2)
    finetuned_weights_path = config["model"]["finetuned_weights_path"]
    device = config.get("device", "auto")

    predictor = SentimentPredictor(
        model_id=model_id,
        num_labels=num_labels,
        finetuned_weights_path=finetuned_weights_path,
        device=device
    )

    result = predictor.predict(
        args.text,
        save_json=False
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()