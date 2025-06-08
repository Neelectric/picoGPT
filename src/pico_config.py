from dataclasses import dataclass

@dataclass
class picoConfig:
    """Class for loading and saving hyperparameters for picoGPT training and inference."""
    context_size: int = 512
    vocab_size: int = 100352
    n_layers: int = 12
    n_heads: int = 12
    d_hidden: int = 768
    
    def load_from_file(self, file_path: str) -> None:
        """Loads a picoConfig from the specified YAML file."""
        raise NotImplementedError
    
    def save_to_file(self, file_path: str) -> None:
        """Saves the current picoConfig as a YAML file to the provided filepath."""
        raise NotImplementedError