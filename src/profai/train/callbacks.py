from typing import Optional

class Callback:
    """Base class for all callbacks."""
    def on_epoch_end(self, trainer, epoch: int):
        pass

    def on_step_end(self, trainer, step: int, loss: float, mode: str):
        pass

    def on_validation_end(self, trainer, step: int, val_loss: float):
        pass

class EarlyStopping(Callback):
    """Early stopping callback.
    
    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_validation_end(self, trainer, step: int, val_loss: float):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at step {step}")
                self.should_stop = True

class LossLogger(Callback):
    """Simple loss logger callback.
    """
    def on_step_end(self, trainer, step: int, loss: float, mode: str):
        print(f"[Step {step}] {mode} loss: {loss:.4f}")