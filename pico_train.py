
import wandb
import fire
from src.pico_gpt import picoGPT
from src.pico_config import picoConfig

def pico_train(file_path: str = "recipes/default.yml"):
    """Trains a PicoGPT model."""
    ### 1. Dataloader Setup

    ### 2. Load hyperparameters from config
    config = picoConfig()
    config.load_from_file(file_path=file_path)

    ### 3. Model and data-parallel setup

    ### 3. wandb, viz code

    ### 4. Model training loop




if __name__ == "__main__":
    fire.Fire()