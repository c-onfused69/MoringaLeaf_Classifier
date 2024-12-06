import os
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MODEL_PATH = "E:/AAA/MoringaLeaf_Classifier/models/moringaleaf_model.h5"
TRAIN_DIR = "E:/AAA/MoringaLeaf_Classifier/dataset/traning_set"
VALIDATION_DIR = "E:/AAA/MoringaLeaf_Classifier/dataset/validation_set"
TEST_DIR = "E:/AAA/MoringaLeaf_Classifier/dataset/testing_set"

# Helper functions for the app
def train_model():
    try:
        os.system("python main.py")  # Run the training script
        messagebox.showinfo("Training", "Model training completed!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during training: {e}")

def test_model():
    try:
        # Ask user to select an image
        image_path = filedialog.askopenfilename(
            title="Select a Leaf Image",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")],
        )
        if not image_path:
            return

        # Load the model
        model = load_model(MODEL_PATH)

        # Preprocess the image
        image_size = (255, 255)
        image = load_img(image_path, target_size=image_size)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make predictions
        prediction = model.predict(image_array)
        class_labels = {0: "Diseased", 1: "Healthy"}
        predicted_class = class_labels[1] if prediction[0] > 0.5 else class_labels[0]

        messagebox.showinfo("Prediction", f"The model predicts: {predicted_class}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during testing: {e}")

def evaluate_model():
    try:
        # Load the model
        model = load_model(MODEL_PATH)

        # Prepare the test data generator
        image_size = (255, 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR,
            target_size=image_size,
            batch_size=32,
            class_mode="binary",
            shuffle=False,
        )

        # Make predictions
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = (predictions > 0.5).astype("int32")
        true_classes = test_generator.classes

        # Generate metrics
        cm = confusion_matrix(true_classes, predicted_classes)
        report = classification_report(
            true_classes, predicted_classes, target_names=["Diseased", "Healthy"]
        )
        accuracy = np.sum(predicted_classes.flatten() == true_classes) / len(true_classes)

        # Display metrics
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        print(f"\nAccuracy: {accuracy * 100:.2f}%")

        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Diseased", "Healthy"],
            yticklabels=["Diseased", "Healthy"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during evaluation: {e}")

# GUI design
def create_gui():
    root = tk.Tk()
    root.title("Moringa Leaf Classification App")
    root.geometry("400x300")

    # Labels
    Label(root, text="Moringa Leaf Classification", font=("Arial", 16)).pack(pady=10)

    # Buttons
    Button(root, text="Train Model", font=("Arial", 12), command=train_model).pack(pady=10)
    Button(root, text="Test Model", font=("Arial", 12), command=test_model).pack(pady=10)
    Button(root, text="Evaluate Model", font=("Arial", 12), command=evaluate_model).pack(pady=10)

    # Exit button
    Button(root, text="Exit", font=("Arial", 12), command=root.quit).pack(pady=10)

    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    create_gui()
