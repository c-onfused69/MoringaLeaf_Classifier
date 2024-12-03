import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

# Suppress TensorFlow verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
model_path = "E:/AAA/MoringaLeaf_Classifier/models/moringaleaf_model.h5"
image_path = "E:/AAA/MoringaLeaf_Classifier/ttest/diseased/PXL_20241128_015257680.jpg"  # Replace with the path to your test image

# Load the trained model
model = load_model(model_path)

# Preprocess the image
image_size = (255, 255)  # Must match the size used during training
image = load_img(image_path, target_size=image_size)  # Load and resize the image
image_array = img_to_array(image)  # Convert the image to a numpy array
image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

# Make a prediction
prediction = model.predict(image_array)
class_labels = {0: "Diseased", 1: "Healthy"}  # Reverse the class labels if needed
predicted_class = class_labels[1] if prediction[0] > 0.5 else class_labels[0]

# Display the result
print(f"The model predicts that the leaf is: {predicted_class}")
