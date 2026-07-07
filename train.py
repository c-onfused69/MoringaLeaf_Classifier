import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger, TensorBoard # type: ignore

from config import Config
from utils.seed import set_seed
from data.dataset import load_image_paths_and_labels, get_kfold_splits, create_tf_dataset
from models.architectures import build_model, unfreeze_top_layers

def train_fold(fold_num, train_idx, val_idx, image_paths, labels, model_name, args):
    """
    Trains the model for a single fold.
    """
    print(f"\n--- Starting Fold {fold_num + 1}/{Config.K_FOLDS} ---")
    
    # Determine input shape based on model
    input_shape = Config.INPUT_SHAPE_INCEPTION if model_name.lower() == "inception_v3" else Config.INPUT_SHAPE_DEFAULT
    
    # Split data
    train_paths, train_labels = image_paths[train_idx], labels[train_idx]
    val_paths, val_labels = image_paths[val_idx], labels[val_idx]
    
    # Create tf.data datasets
    train_dataset = create_tf_dataset(
        train_paths, train_labels, 
        target_size=input_shape[:2], 
        batch_size=Config.BATCH_SIZE, 
        is_training=True, 
        config=Config
    )
    val_dataset = create_tf_dataset(
        val_paths, val_labels, 
        target_size=input_shape[:2], 
        batch_size=Config.BATCH_SIZE, 
        is_training=False, 
        config=Config
    )
    
    # Build Model
    model, base_model, loss_fn = build_model(model_name, input_shape, Config.NUM_CLASSES)
    
    # Output directory for this fold
    fold_dir = os.path.join(Config.OUTPUTS_DIR, model_name, f"fold_{fold_num}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(os.path.join(fold_dir, "best_model.keras"), monitor='val_loss', save_best_only=True),
        CSVLogger(os.path.join(fold_dir, "training_log.csv")),
        TensorBoard(log_dir=os.path.join(fold_dir, "logs"))
    ]
    
    if args.quick_test:
        epochs_phase1 = 1
        epochs_phase2 = 1
    else:
        epochs_phase1 = Config.PHASE1_EPOCHS
        epochs_phase2 = Config.PHASE2_EPOCHS
    
    # ==========================================
    # Phase 1: Train Head only (frozen backbone)
    # ==========================================
    print("Phase 1: Training head...")
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE_PHASE1), 
        loss=loss_fn, 
        metrics=["accuracy"]
    )
    
    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_phase1,
        callbacks=callbacks
    )
    
    # ==========================================
    # Phase 2: Fine-tuning (unfrozen top layers)
    # ==========================================
    print("Phase 2: Fine-tuning top layers...")
    base_model = unfreeze_top_layers(base_model, unfreeze_fraction=0.3)
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE_PHASE2), 
        loss=loss_fn, 
        metrics=["accuracy"]
    )
    
    # Add EarlyStopping and ReduceLR for fine-tuning
    callbacks.extend([
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    ])
    
    history_phase2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_phase2,
        callbacks=callbacks
    )
    
    # Save final history combined
    return history_phase1, history_phase2

def main():
    parser = argparse.ArgumentParser(description="Train Moringa Leaf Classification Model")
    parser.add_argument("--model", type=str, default="inception_v3", 
                        choices=["inception_v3", "resnet50", "densenet121", "efficientnetb0", "mobilenetv3", "vit"],
                        help="Model architecture to train")
    parser.add_argument("--folds", type=int, default=0, 
                        help="Number of folds to run (0 means all)")
    parser.add_argument("--quick-test", action="store_true", help="Run a fast 1-epoch test")
    
    args = parser.parse_args()
    
    # Set reproducibility
    set_seed(Config.SEED)
    
    # Load dataset paths
    data_dirs = [Config.TRAIN_DIR, Config.VAL_DIR] # Combine train and val for cross-validation
    image_paths, labels = load_image_paths_and_labels(data_dirs, Config.CLASS_NAMES)
    
    if len(image_paths) == 0:
        print("No images found! Please check your dataset directory structure.")
        return
        
    print(f"Loaded {len(image_paths)} images across {Config.NUM_CLASSES} classes.")
    
    # Get k-fold splits
    splits = get_kfold_splits(image_paths, labels, k_folds=Config.K_FOLDS, seed=Config.SEED)
    
    num_folds_to_run = args.folds if args.folds > 0 else Config.K_FOLDS
    
    for fold_num, (train_idx, val_idx) in enumerate(splits):
        if fold_num >= num_folds_to_run:
            break
            
        train_fold(fold_num, train_idx, val_idx, image_paths, labels, args.model, args)
        
    print("Training complete!")

if __name__ == "__main__":
    main()
