import yaml

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config