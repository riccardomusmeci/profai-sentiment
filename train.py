import argparse
import yaml
import os

from datasets import load_dataset
from torch.utils.data import DataLoader

from profai.utils.config import load_config
from profai.data.sentiment import SentimentDataset
from profai.models import create_model_and_tokenizer
from profai.train.utils import create_loss, create_optimizer, create_scheduler
from profai.train.trainer import Trainer
from profai.train.callbacks import EarlyStopping, LossLogger


def main():
    parser = argparse.ArgumentParser(description="ProfAI Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    # 1. Load dataset
    dataset_name = config["dataset"]["name"]
    max_train_samples = config["dataset"].get("max_train_samples", 512)
    max_val_samples = config["dataset"].get("max_val_samples", 128)
    batch_size = config["training"].get("batch_size", 8)
    seed = config["training"].get("seed", 42)

    raw_dataset = load_dataset(dataset_name)
    train_data = raw_dataset["train"].shuffle(seed=seed).select(range(max_train_samples))
    val_data = raw_dataset["test"].shuffle(seed=seed).select(range(max_val_samples))

    train_samples = [{"text": x["text"], "label": x["label"]} for x in train_data]
    val_samples = [{"text": x["text"], "label": x["label"]} for x in val_data]

    # 2. Model and tokenizer
    model_id = config["model"]["model_id"]
    num_labels = config["model"].get("num_labels", 2)
    model, tokenizer = create_model_and_tokenizer(model_id, num_labels)

    # 3. Datasets and dataloaders
    max_length = config["model"].get("max_length", 128)
    train_dataset = SentimentDataset(train_samples, tokenizer, max_length=max_length)
    val_dataset = SentimentDataset(val_samples, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 4. Optimizer, scheduler, loss
    optimizer_cfg = config["optimizer"]
    optimizer = create_optimizer(
        optimizer_cfg.get("type", "adamw"),
        model.parameters(),
        lr=optimizer_cfg.get("lr", 2e-5)
    )
    scheduler_cfg = config.get("scheduler", {})
    scheduler = create_scheduler(
        scheduler_cfg.get("type", "none"),
        optimizer,
        **{k: v for k, v in scheduler_cfg.items() if k != "type"}
    )
    loss_fn = create_loss(config["loss"].get("type", "cross_entropy"))

    # 5. Callbacks
    early_stopping_cfg = config.get("early_stopping", {})
    callbacks = [
        EarlyStopping(
            patience=early_stopping_cfg.get("patience", 2),
            min_delta=early_stopping_cfg.get("min_delta", 0.01)
        ),
        LossLogger()
    ]

    # 6. Trainer
    output_dir = config["training"].get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=config["training"].get("device", "auto"),
        output_dir=output_dir,
        save_every=config["training"].get("save_every", 16),
        log_every=config["training"].get("log_every", 16),
        validate_every=config["training"].get("validate_every", 64),
        max_epochs=config["training"].get("max_epochs", 5),
        callbacks=callbacks
    )

    # 7. Train
    trainer.train()

if __name__ == "__main__":
    main()