import os
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress TensorFlow verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
model_path = "E:/AAA/MoringaLeaf_Classifier/models/moringaleaf_model.h5"
test_dir = "E:/AAA/MoringaLeaf_Classifier/ttest"  # Path to the test directory

# Load the trained model
model = load_model(model_path)

# Prepare the test data generator
image_size = (255, 255)  # Must match the size used during training
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=32,
    class_mode="binary",
    shuffle=False  # Important to not shuffle the test data for accurate metrics
)

# Make predictions on the test data
predictions = model.predict(test_generator, verbose=1)

# Convert predictions to binary values (0 or 1)
predicted_classes = (predictions > 0.5).astype("int32")

# Get the true classes from the generator
true_classes = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix using Seaborn
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

# Classification Report (Precision, Recall, F1-Score)
report = classification_report(
    true_classes, predicted_classes, target_names=["Healthy", "Diseased"]
)
print("Classification Report:")
print(report)

# Corrected Accuracy Calculation
accuracy = np.sum(predicted_classes.flatten() == true_classes) / len(true_classes)
print(f"Accuracy: {accuracy * 100:.2f}%")
