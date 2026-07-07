import os
import argparse
import pandas as pd
from train import train_fold
from config import Config
from data.dataset import load_image_paths_and_labels, get_kfold_splits

def run_ablation(experiment_name, args):
    """
    Runs a specific ablation experiment.
    """
    print(f"\n========== Running Ablation: {experiment_name} ==========")
    
    # Base configuration overrides for experiments
    original_config = {
        'ROTATION_RANGE': Config.ROTATION_RANGE,
        'PHASE1_EPOCHS': Config.PHASE1_EPOCHS,
        'PHASE2_EPOCHS': Config.PHASE2_EPOCHS,
        'LEARNING_RATE_PHASE2': Config.LEARNING_RATE_PHASE2
    }
    
    experiments = {}
    
    if experiment_name == "augmentation":
        experiments = {
            "no_aug": {"ROTATION_RANGE": 0, "WIDTH_SHIFT_RANGE": 0, "HEIGHT_SHIFT_RANGE": 0},
            "mild_aug": {"ROTATION_RANGE": 15, "WIDTH_SHIFT_RANGE": 0.1, "HEIGHT_SHIFT_RANGE": 0.1},
            "heavy_aug": {"ROTATION_RANGE": 45, "WIDTH_SHIFT_RANGE": 0.3, "HEIGHT_SHIFT_RANGE": 0.3}
        }
    elif experiment_name == "freezing":
        # Handled in modified build_model/train logic, but for simplicity we simulate by changing epochs
        experiments = {
            "freeze_all": {"PHASE2_EPOCHS": 0}, 
            "fine_tune_long": {"PHASE2_EPOCHS": 60}
        }
        
    # Load dataset
    image_paths, labels = load_image_paths_and_labels([Config.TRAIN_DIR, Config.VAL_DIR], Config.CLASS_NAMES)
    splits = get_kfold_splits(image_paths, labels, k_folds=Config.K_FOLDS, seed=Config.SEED)
    
    # We only use fold 0 for quick ablation studies unless specified
    train_idx, val_idx = splits[0]
    
    results = []
    
    for exp_id, params in experiments.items():
        print(f"\n--- Testing variant: {exp_id} ---")
        
        # Apply overrides
        for k, v in params.items():
            setattr(Config, k, v)
            
        # Run training
        hist1, hist2 = train_fold(
            fold_num=0, 
            train_idx=train_idx, 
            val_idx=val_idx, 
            image_paths=image_paths, 
            labels=labels, 
            model_name=args.model, 
            args=args
        )
        
        # Extract best val_accuracy
        if hist2:
            best_val_acc = max(hist2.history['val_accuracy'])
        else:
            best_val_acc = max(hist1.history['val_accuracy'])
            
        results.append({
            "Experiment": experiment_name,
            "Variant": exp_id,
            "Best_Val_Accuracy": best_val_acc
        })
        
        # Restore original config
        for k, v in original_config.items():
            setattr(Config, k, v)
            
    # Save results
    df = pd.DataFrame(results)
    out_dir = os.path.join(Config.OUTPUTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{experiment_name}_results.csv"), index=False)
    print(f"\nAblation results saved to {out_dir}")
    print(df)

def main():
    parser = argparse.ArgumentParser(description="Run Ablation Studies")
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["augmentation", "freezing"],
                        help="Which ablation study to run")
    parser.add_argument("--model", type=str, default="inception_v3")
    parser.add_argument("--quick-test", action="store_true", help="Run a fast 1-epoch test")
    
    args = parser.parse_args()
    
    run_ablation(args.experiment, args)

if __name__ == "__main__":
    main()
