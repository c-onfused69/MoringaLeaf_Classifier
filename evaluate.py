import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score, roc_curve

from config import Config
from data.dataset import load_image_paths_and_labels, create_tf_dataset
from visualization.plots import plot_confusion_matrix, plot_roc_curve

def evaluate_model(model_path, test_dataset, true_labels, class_names):
    """
    Evaluates a saved model on the test dataset.
    """
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return None
        
    model = tf.keras.models.load_model(model_path)
    
    print(f"Evaluating {model_path}...")
    
    print("\nGenerating predictions...")
    predictions = model.predict(test_dataset, verbose=0)
    
    if predictions.shape[1] == 1:
        # Binary classification (sigmoid)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        pred_probs = predictions.flatten()
    else:
        # Multi-class (softmax)
        predicted_classes = np.argmax(predictions, axis=1)
        # Using probs of the positive class for binary ROC, or adapt for multi-class
        pred_probs = predictions[:, 1] if len(class_names) == 2 else predictions
        
    # Metrics
    cm = confusion_matrix(true_labels, predicted_classes)
    report_dict = classification_report(true_labels, predicted_classes, target_names=class_names, output_dict=True)
    kappa = cohen_kappa_score(true_labels, predicted_classes)
    
    # ROC AUC
    try:
        if len(class_names) == 2:
            auc = roc_auc_score(true_labels, pred_probs)
        else:
            auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
    except Exception as e:
        print(f"Could not calculate ROC AUC: {e}")
        auc = None
        
    metrics = {
        'accuracy': report_dict['accuracy'],
        'macro_f1': report_dict['macro avg']['f1-score'],
        'weighted_f1': report_dict['weighted avg']['f1-score'],
        'kappa': kappa,
        'auc': auc,
        'report': report_dict,
        'cm': cm,
        'pred_probs': pred_probs,
        'true_labels': true_labels
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models")
    parser.add_argument("--model-name", type=str, default="inception_v3", help="Model name to evaluate")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to evaluate")
    parser.add_argument("--all-folds", action="store_true", help="Evaluate all folds and compute mean/std")
    
    args = parser.parse_args()
    
    # Setup test dataset
    image_paths, true_labels = load_image_paths_and_labels([Config.TEST_DIR], Config.CLASS_NAMES)
    
    if len(image_paths) == 0:
        print("No images found in test directory!")
        return
        
    input_shape = Config.INPUT_SHAPE_INCEPTION if args.model_name.lower() == "inception_v3" else Config.INPUT_SHAPE_DEFAULT
    
    test_dataset = create_tf_dataset(
        image_paths, true_labels, 
        target_size=input_shape[:2], 
        batch_size=Config.BATCH_SIZE, 
        is_training=False, 
        config=None
    )
    
    if args.all_folds:
        all_metrics = []
        for f in range(Config.K_FOLDS):
            model_path = os.path.join(Config.OUTPUTS_DIR, args.model_name, f"fold_{f}", "best_model.keras")
            metrics = evaluate_model(model_path, test_dataset, true_labels, Config.CLASS_NAMES)
            if metrics:
                all_metrics.append(metrics)
                
        if all_metrics:
            accuracies = [m['accuracy'] for m in all_metrics]
            f1_scores = [m['macro_f1'] for m in all_metrics]
            
            print(f"\n--- Results across {len(all_metrics)} folds for {args.model_name} ---")
            print(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            print(f"Macro F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
            
    else:
        model_path = os.path.join(Config.OUTPUTS_DIR, args.model_name, f"fold_{args.fold}", "best_model.keras")
        metrics = evaluate_model(model_path, test_dataset, true_labels, Config.CLASS_NAMES)
        
        if metrics:
            print(f"\n--- Results for {args.model_name} (Fold {args.fold}) ---")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Macro F1: {metrics['macro_f1']:.4f}")
            print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
            if metrics['auc']:
                print(f"ROC AUC: {metrics['auc']:.4f}")
            
            print("\nClassification Report:")
            print(classification_report(metrics['true_labels'], (metrics['pred_probs'] > 0.5).astype(int), target_names=Config.CLASS_NAMES))
            
            # Save plots and metrics
            out_dir = os.path.join(Config.OUTPUTS_DIR, args.model_name, f"fold_{args.fold}")
            plot_confusion_matrix(metrics['cm'], Config.CLASS_NAMES, os.path.join(out_dir, "confusion_matrix.png"))
            if len(Config.CLASS_NAMES) == 2:
                plot_roc_curve(metrics['true_labels'], metrics['pred_probs'], os.path.join(out_dir, "roc_curve.png"))
                
            import json
            metrics_to_save = {
                'accuracy': float(metrics['accuracy']),
                'macro_f1': float(metrics['macro_f1']),
                'weighted_f1': float(metrics['weighted_f1']),
                'kappa': float(metrics['kappa']),
                'auc': float(metrics['auc']) if metrics['auc'] else None,
                'report': metrics['report'],
                'cm': metrics['cm'].tolist()
            }
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics_to_save, f, indent=4)

if __name__ == "__main__":
    main()
