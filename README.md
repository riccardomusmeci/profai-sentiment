# ProfAI - Professional AI Library

A Python library for training and inference of sentiment analysis models using HuggingFace transformer models.

## Features

- 🚀 **Easy to use**: Training and inference with simple command-line commands
- 🔧 **Configurable**: Configuration via YAML files for maximum flexibility
- 🏗️ **Modular architecture**: Reusable components for data loading, training, and inference
- 💾 **Model management**: Automatic checkpoint saving during training
- 📊 **Monitoring**: Callbacks for early stopping and metrics logging
- 🎯 **Optimized**: Support for GPU, CPU, and Apple Silicon (MPS)

## Installation

```bash
git clone https://github.com/riccardomusmeci/profai-sentiment
cd profai-sentiment
pip install -e .
```

## Usage

### 1. Training

Training is performed via the `train.py` script using a YAML configuration file.

#### Training Configuration

Create a configuration file (example `config/train.yaml`):

```yaml
model:
  model_id: distilbert-base-uncased  # HuggingFace model to use
  num_labels: 2                      # Number of classes (2 for binary sentiment)
  max_length: 128                    # Maximum sequence length

dataset:
  name: imdb                         # HuggingFace dataset name
  max_train_samples: 1024            # Maximum number of training samples
  max_val_samples: 256               # Maximum number of validation samples

optimizer:
  type: adamw                        # Optimizer type
  lr: 0.00002                        # Learning rate

scheduler:
  type: none                         # Scheduler type (none, linear, cosine)

loss:
  type: cross_entropy                # Loss function

early_stopping:
  patience: 2                        # Number of epochs without improvement
  min_delta: 0.01                    # Minimum improvement required

training:
  batch_size: 8                      # Batch size
  output_dir: ./ckpt                 # Output directory for checkpoints
  save_every: 16                     # Save checkpoint every N steps
  log_every: 16                      # Log metrics every N steps
  validate_every: 64                 # Validate every N steps
  max_epochs: 5                      # Maximum number of epochs
  device: auto                       # Device (auto, cpu, cuda, mps)
  seed: 42                           # Seed for reproducibility
```

#### Running Training

```bash
python train.py --config config/train.yaml
```

During training, the following will happen:
- Data will be loaded from the specified dataset
- Dataloaders will be created with automatic tokenization
- Checkpoints will be saved in the `output_dir` directory
- Early stopping and metrics logging will be applied
- Training and validation metrics will be printed

### 2. Inference

Inference is performed via the `inference.py` script to predict sentiment of individual texts.

#### Inference Configuration

Create a configuration file (example `config/predict.yaml`):

```yaml
model:
  model_id: distilbert-base-uncased                    # Same model used for training
  num_labels: 2                                        # Number of classes
  finetuned_weights_path: ./ckpt/checkpoint_256.pt     # Path to fine-tuned weights

device: auto  # Device to use
```

#### Running Inference

```bash
# Prediction with default configuration
python inference.py --text "This movie is absolutely fantastic!"

# Prediction with custom configuration
python inference.py --config config/predict.yaml --text "This movie is terrible"
```

#### Inference Output

The output is in JSON format and includes:

```json
{
  "text": "This movie is absolutely fantastic!",
  "prediction": 1,
  "probabilities": [0.0766, 0.9234],
}
```
