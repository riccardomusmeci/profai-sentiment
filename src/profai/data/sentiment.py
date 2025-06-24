import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any

class SentimentDataset(Dataset):
    
    VALID_PADDING = ["max_length", "longest", True, False]
    
    def __init__(
        self, 
        data: List[Dict[str, Any]], 
        tokenizer: AutoTokenizer, 
        max_length: int = 256, 
        text_col: str = "text", 
        label_col: str = "label",
        truncation: bool = True,
        padding: str = "max_length"
    ):
        """
        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing the dataset.
            tokenizer (AutoTokenizer): tokenizer
            max_length (int): max length of the tokenized sequences. Default is 256.
            text_col (str): text column name in the dataset. Default is "text".
            label_col (str): label column name in the dataset. Default is "label".
            truncation (bool): whether to truncate sequences longer than max_length. Default is True.
            padding (str): padding strategy. Valid values are "max_length", "longest",
        """
        
        assert padding in self.VALID_PADDING, f"Invalid padding parameter: {padding}. Valid values are: {self.VALID_PADDING}"        
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_col = text_col
        self.label_col = label_col
        self.truncation = truncation
        self.padding = padding

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        text = item[self.text_col]
        label = item[self.label_col]

        encoding = self.tokenizer(
            text,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Rimuovo la dimensione batch aggiunta da return_tensors="pt"
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return {
            **encoding,
            "labels": torch.tensor(label, dtype=torch.long)
        }