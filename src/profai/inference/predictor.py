from typing import List, Union, Optional
import torch
import json
import os
from profai.models import create_model_and_tokenizer
from profai.utils.device import get_device

class SentimentPredictor:
    """Class for sentiment inference on single or multiple texts.
    """

    def __init__(
        self,
        model_id: str,
        num_labels: int,
        finetuned_weights_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Args:
            model_id (str): HuggingFace model identifier.
            num_labels (int): Number of output classes.
            finetuned_weights_path (Optional[str]): Path to fine-tuned weights.
            device (str): Device to use ("auto", "cpu", "cuda", "mps").
        """
        self.model, self.tokenizer = create_model_and_tokenizer(
            model_id=model_id,
            num_labels=num_labels,
            finetuned_weights_path=finetuned_weights_path
        )
        self.device = get_device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        texts: Union[str, List[str]],
        save_json: bool = False,
        output_dir: Optional[str] = None
    ) -> Union[dict, List[dict]]:
        """
        Predict sentiment for a single text or a list of texts.

        Args:
            texts (str or List[str]): Input text(s).
            save_json (bool): If True, save each prediction as a JSON file.
            output_dir (Optional[str]): Directory to save JSON files.

        Returns:
            dict or List[dict]: Prediction results.
        """
        single_input = False
        if isinstance(texts, str):
            texts = [texts]
            single_input = True

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        outputs = self.model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().tolist()
        probs = probs.cpu().tolist()

        results = []
        for i, text in enumerate(texts):
            result = {
                "text": text,
                "prediction": preds[i],
                "probabilities": probs[i]
            }
            results.append(result)
            if save_json and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"prediction_{i}.json")
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)

        if single_input:
            if save_json and output_dir:
                # Save also single file with a fixed name
                out_path = os.path.join(output_dir, "prediction_single.json")
                with open(out_path, "w") as f:
                    json.dump(results[0], f, indent=2)
            return results[0]
        return results