from profai.utils.device import get_device
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, Callable, List
from profai.train.callbacks import Callback
import json

class Trainer:
    """
    Custom Trainer for NLP models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[Any] = None,
        device: str = "auto",
        output_dir: str = "./outputs",
        save_every: int = 100,
        log_every: int = 10,
        validate_every: int = 100,
        max_epochs: int = 3,
        callbacks: Optional[List[Callback]] = None,
    ):
        """
        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            loss_fn: Loss function.
            train_loader: DataLoader for training.
            val_loader: DataLoader for validation.
            scheduler: Learning rate scheduler.
            device: Device to use ("auto", "cpu", "cuda", "mps").
            output_dir: Directory to save checkpoints.
            save_every: Save checkpoint every N iterations.
            log_every: Log loss every N iterations.
            validate_every: Run validation every N iterations.
            max_epochs: Number of epochs.
            callbacks: Optional callbacks.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = get_device(device)
        self.output_dir = output_dir
        self.save_every = save_every
        self.log_every = log_every
        self.validate_every = validate_every
        self.max_epochs = max_epochs
        self.callbacks = callbacks or []
        self.train_loss_history = []
        self.val_loss_history = []

        os.makedirs(self.output_dir, exist_ok=True)
        self.model.to(self.device)

    def save_checkpoint(self, step: int) -> None:
        """Save model, optimizer, scheduler state.
        
        Args:
            step (int): Current training step.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, os.path.join(self.output_dir, f"checkpoint_{step}.pt"))

    def train(self):
        """Main training loop.
        """
        global_step = 0
        for epoch in range(self.max_epochs):
            self.model.train()
            running_loss = 0.0
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            for batch in self.train_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                running_loss += loss.item()
                global_step += 1

                if global_step % self.log_every == 0:
                    avg_loss = running_loss / self.log_every
                    self.train_loss_history.append((global_step, avg_loss))
                    for callback in self.callbacks:
                        callback.on_step_end(self, global_step, avg_loss, "train")
                    running_loss = 0.0

                if global_step % self.save_every == 0:
                    self.save_checkpoint(global_step)

                if self.val_loader and (global_step % self.validate_every == 0):
                    val_loss, val_accuracy = self.validate()
                    self.val_loss_history.append({"step": global_step, "loss": val_loss, "accuracy": val_accuracy})
                    for callback in self.callbacks:
                        callback.on_validation_end(self, global_step, val_loss)
                    
                    # Early stopping check
                    if any(getattr(cb, "should_stop", False) for cb in self.callbacks):
                        print("Early stopping triggered. Exiting training loop.")
                        self.save_loss_history()
                        return

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch)
                
        self.save_loss_history()

    def save_loss_history(self) -> None:
        """Save training and validation loss history."""
        loss_history = {
            "train": self.train_loss_history,
            "val": self.val_loss_history,
        }
        with open(os.path.join(self.output_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)
        print(f"Loss history saved to {self.output_dir}/loss_history.json")

    def validate(self) -> float:
        """Run validation loop.
        
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(self.device)
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs.logits, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                count += 1
        avg_loss = total_loss / max(count, 1)
        accuracy = correct / total if total > 0 else 0.0
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy