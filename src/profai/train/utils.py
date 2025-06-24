from typing import Any, Dict, List, Optional
import torch
import torch.optim as optim
import torch.nn as nn

OPTIMIZER_REGISTRY = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}

SCHEDULER_REGISTRY = {
    "steplr": optim.lr_scheduler.StepLR,
    "reducelronplateau": optim.lr_scheduler.ReduceLROnPlateau,
    "lambdalr": optim.lr_scheduler.LambdaLR,
    "cosineannealinglr": optim.lr_scheduler.CosineAnnealingLR,
    "none": None,  # For no scheduler
}

LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
}

def create_optimizer(
    optimizer_name: str,
    model_params,
    lr: float = 1e-4,
    optimizer_kwargs: Dict[str, Any] = None
) -> optim.Optimizer:
    """Create a PyTorch optimizer.
    
    Args:
        optimizer_name (str): Name of the optimizer.
        model_params: Parameters of the model to optimize.
        lr (float): Learning rate. Default is 1e-4.
        optimizer_kwargs (Dict[str, Any], optional): Additional arguments for the optimizer.
        
    Returns:
        optim.Optimizer: The created optimizer.
    """
    optimizer_kwargs = optimizer_kwargs or {}
    if optimizer_name.lower() not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return OPTIMIZER_REGISTRY[optimizer_name.lower()](model_params, lr=lr, **optimizer_kwargs)

def create_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    scheduler_kwargs: Dict[str, Any] = None
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create a PyTorch learning rate scheduler.
    
    Args:
        scheduler_name (str): Name of the scheduler.
        optimizer (optim.Optimizer): The optimizer to attach the scheduler to.
        scheduler_kwargs (Dict[str, Any], optional): Additional arguments for the scheduler.
        
    Returns:
        Optional[optim.lr_scheduler._LRScheduler]: The created scheduler or None if no scheduler
    """
    scheduler_kwargs = scheduler_kwargs or {}
    if scheduler_name.lower() == "none":
        return None
    if scheduler_name.lower() not in SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    return SCHEDULER_REGISTRY[scheduler_name.lower()](optimizer, **scheduler_kwargs)

def create_loss(
    loss_name: str,
    loss_kwargs: Dict[str, Any] = None
) -> nn.Module:
    """Create a PyTorch loss function.
    
    Args:
        loss_name (str): Name of the loss function.
        loss_kwargs (Dict[str, Any], optional): Additional arguments for the loss function.
        
    Returns:
        nn.Module: The loss function.
    """
    loss_kwargs = loss_kwargs or {}
    losses = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce": nn.BCEWithLogitsLoss,
        "mse": nn.MSELoss,
    }
    if loss_name.lower() not in losses:
        raise ValueError(f"Unknown loss: {loss_name}")
    return losses[loss_name.lower()](**loss_kwargs)