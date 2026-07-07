import os
import yaml

class Config:
    # Base paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    STATIC_DIR = os.path.join(BASE_DIR, "static")
    UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
    
    # Dataset splits
    TRAIN_DIR = os.path.join(DATASET_DIR, "traning_set")  # Using existing typo
    VAL_DIR = os.path.join(DATASET_DIR, "validation_set")
    TEST_DIR = os.path.join(DATASET_DIR, "testing_set")
    
    # Model parameters
    INPUT_SHAPE_DEFAULT = (224, 224, 3)
    INPUT_SHAPE_INCEPTION = (299, 299, 3)
    NUM_CLASSES = 2
    CLASS_NAMES = ["Diseased", "Healthy"]
    
    # Training parameters
    BATCH_SIZE = 32
    K_FOLDS = 5
    PHASE1_EPOCHS = 10
    PHASE2_EPOCHS = 40
    LEARNING_RATE_PHASE1 = 1e-3
    LEARNING_RATE_PHASE2 = 1e-4
    SEED = 42
    
    # Augmentation
    ROTATION_RANGE = 30
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    BRIGHTNESS_RANGE = [0.8, 1.2]

    @classmethod
    def setup_dirs(cls):
        """Create necessary directories if they don't exist."""
        for d in [cls.MODELS_DIR, cls.OUTPUTS_DIR, cls.STATIC_DIR, cls.UPLOADS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def load_from_yaml(cls, yaml_path):
        """Load configuration from a YAML file (optional override)."""
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                overrides = yaml.safe_load(f)
                if overrides:
                    for k, v in overrides.items():
                        if hasattr(cls, k):
                            setattr(cls, k, v)

# Ensure directories exist
Config.setup_dirs()
