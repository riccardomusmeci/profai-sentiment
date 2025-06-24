from typing import Any, Dict, Optional, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
import torch
import os

logger = logging.getLogger(__name__)

def create_model_and_tokenizer(
    model_id: str,
    num_labels: int,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    finetuned_weights_path: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Loads a HuggingFace model and tokenizer for sequence classification.
    Optionally loads fine-tuned weights.

    Args:
        model_id (str): HuggingFace model identifier.
        num_labels (int): Number of output classes.
        cache_dir (Optional[str]): Directory to cache the model/tokenizer.
        revision (Optional[str]): Model revision/version.
        tokenizer_kwargs (Optional[Dict]): Additional kwargs for tokenizer.
        model_kwargs (Optional[Dict]): Additional kwargs for model.
        finetuned_weights_path (Optional[str]): Path to fine-tuned weights.

    Returns:
        Tuple[Any, Any]: (model, tokenizer)

    Raises:
        RuntimeError: If model or tokenizer cannot be loaded.
    """
    tokenizer_kwargs = tokenizer_kwargs or {}
    model_kwargs = model_kwargs or {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            revision=revision,
            **tokenizer_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer for '{model_id}': {e}")
        raise RuntimeError(f"Failed to load tokenizer for '{model_id}': {e}")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            cache_dir=cache_dir,
            revision=revision,
            **model_kwargs
        )
        if finetuned_weights_path and os.path.isfile(finetuned_weights_path):
            state = torch.load(finetuned_weights_path, map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            logger.info(f"Loaded fine-tuned weights from {finetuned_weights_path}")
    except Exception as e:
        logger.error(f"Failed to load model for '{model_id}': {e}")
        raise RuntimeError(f"Failed to load model for '{model_id}': {e}")

    return model, tokenizer