import os
from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Suppress TensorFlow verbose logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2 suppresses info; 3 suppresses warnings and errors

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "secret_key"

# Paths
MODEL_PATH = "E:/AAA/MoringaLeaf_Classifier/models/moringaleaf_model.h5"
TRAIN_SCRIPT = "E:/AAA/MoringaLeaf_Classifier/scripts/main.py"
TEST_DIR = "E:/AAA/MoringaLeaf_Classifier/dataset/testing_set"
# TEST_DIR = "D:/Projects/Rembg/Single leaf-20241013T040529Z-001"
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Helper functions
def preprocess_image(image_path, image_size=(255, 255)):
    """Load and preprocess the image."""
    image = load_img(image_path, target_size=image_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def plot_confusion_matrix(cm, labels):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    save_path = os.path.join(STATIC_FOLDER, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved at {save_path}")

def plot_classification_report(report_dict, labels):
    """Plot and save the classification report as a bar chart."""
    metrics = ["precision", "recall", "f1-score"]
    values = {metric: [report_dict[label][metric] for label in labels] for metric in metrics}

    # Create bar chart
    x = np.arange(len(labels))  # Label indices
    width = 0.25  # Width of bars
    plt.figure(figsize=(8, 6))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * width, values[metric], width, label=metric)

    plt.xticks(x + width, labels)
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.title("Classification Report Metrics")
    plt.ylim(0, 1.1)
    plt.legend(loc="lower right")
    save_path = os.path.join(STATIC_FOLDER, "classification_report.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Classification report chart saved at {save_path}")

# Routes
@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/train", methods=["POST"])
def train():
    """Train the model by calling the main script."""
    try:
        os.system(f"python {TRAIN_SCRIPT}")
        flash("Model training completed successfully!", "success")
    except Exception as e:
        flash(f"Error during training: {e}", "danger")
    return redirect(url_for("home"))

@app.route("/test", methods=["POST"])
def test():
    """Test the model with an uploaded image."""
    try:
        # Save uploaded file
        if "image" not in request.files:
            flash("No file uploaded!", "danger")
            return redirect(url_for("home"))
        file = request.files["image"]
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Load the model
        model = load_model(MODEL_PATH)

        # Preprocess the image
        image_array = preprocess_image(file_path)

        # Make predictions
        prediction = model.predict(image_array)
        class_labels = {0: "Diseased", 1: "Healthy"}
        predicted_class = class_labels[1] if prediction[0] > 0.5 else class_labels[0]

        flash(f"The model predicts: {predicted_class}", "info")
    except Exception as e:
        flash(f"Error during testing: {e}", "danger")
    return redirect(url_for("home"))

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """Evaluate the model on the test dataset."""
    try:
        # Load the model
        model = load_model(MODEL_PATH)

        # Prepare the test data generator
        image_size = (255, 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = test_datagen.flow_from_directory(
            TEST_DIR, target_size=image_size, batch_size=32, class_mode="binary", shuffle=False
        )

        # Make predictions
        predictions = model.predict(test_generator, verbose=1)
        predicted_classes = (predictions > 0.5).astype("int32")
        true_classes = test_generator.classes

        # Confusion matrix and classification report
        cm = confusion_matrix(true_classes, predicted_classes)
        report_dict = classification_report(
            true_classes, predicted_classes, target_names=["Diseased", "Healthy"], output_dict=True
        )
        report = classification_report(true_classes, predicted_classes, target_names=["Diseased", "Healthy"])
        accuracy = np.sum(predicted_classes.flatten() == true_classes) / len(true_classes)

        # Save confusion matrix and classification report charts
        plot_confusion_matrix(cm, ["Diseased", "Healthy"])
        plot_classification_report(report_dict, ["Diseased", "Healthy"])

        # Display results in flash messages
        flash("Evaluation completed!", "success")
        flash(f"Accuracy: {accuracy * 100:.2f}%", "info")
        flash(f"Classification Report:\n{report}", "info")
    except Exception as e:
        flash(f"Error during evaluation: {e}", "danger")
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=False)
