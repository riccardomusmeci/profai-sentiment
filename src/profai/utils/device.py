import torch

def get_device(device: str = "auto") -> torch.device:
    """
    Select the best device available.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    print(f"Using device: {device}")
    return torch.device(device)