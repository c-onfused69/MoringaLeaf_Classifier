# MoringaLeaf_Classifier
A deep learning-based web application for classifying the health of Moringa leaves using an Inception-v4 architecture. This project detects diseases in leaves and provides insights through metrics like accuracy, confusion matrices, and classification reports, visualized with charts for better interpretability.

# Moringa Leaf Classification Application 🌿

A web-based application that classifies Moringa leaves into *Healthy* or *Diseased* categories using a deep learning model. The application is built using Flask and TensorFlow, allowing users to train, test, and evaluate the model directly from the web interface.

---

## **Features**
1. **Train the Model:** Train the deep learning model using pre-labeled datasets.
2. **Test the Model:** Upload a Moringa leaf image to predict whether it is *Healthy* or *Diseased*.
3. **Evaluate the Model:** Analyze model performance with confusion matrices and classification reports.
4. **Visualization:** View confusion matrices and classification metrics for better insights.

---

## **Project Structure**

MoringaLeaf_Classifier/
├── dataset/
│   ├── testing_set/
│   │   ├── diseased/*.png
│   │   └── healthy/*.png
│   ├── training_set/
│   │   ├── diseased/*.png
│   │   └── healthy/*.png
│   └── validation_set/
│       ├── diseased/*.png
│       └── healthy/*.png
│
├── app.py # Main Flask application
|
├── models/ 
│   └── moringaleaf_model.h5 # Pretrained model file
|
├── scripts/ 
│   ├── main.py # Model training script
|   ├── test.py # Model testing script
|   ├── evaluation.py # Model evaluation script
|   └── flux_ui.py # UI for model evaluation using flux
|
├── static/ 
│   ├── confusion_matrix.png # Generated confusion matrix 
│   ├── classification_report.png # Generated classification report metrics chart 
│   └── styles.css # Custom CSS for styling
|
├── templates/
│   └── index.html # HTML template for the web app
|
├── test/
│   └── test_images.png # Test images for model testing
|
├── ttest/ # Directory containing test dataset 
├── uploads/ # Directory for uploaded images (auto-created) 
└── README.md # Documentation file
└── requirements.txt # List of dependencies


---

## **How to Run the Project**

### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.7 or later
- TensorFlow
- Flask
- Required Python packages listed in `requirements.txt` (if provided)

### **2. Clone the Repository**
#### bash
git clone https://github.com/c-onfused69/MoringaLeaf_Classifier.git
cd MoringaLeaf_Classifier

### **3. Install Dependencies**
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
pip install -r requirements.txt

### **4. Prepare the Dataset**
Add training and testing datasets to appropriate directories.
Ensure the test dataset is placed in the ttest/ directory, organized into subdirectories for each class (e.g., Healthy, Diseased).

### **5. Run the Application**
python app.py
Access the web app at http://127.0.0.1:5000/.

## **Usage**

### ***Home Page***
Use the web interface to interact with the app.

### ***Train the Model***
Click the "Train Model" button to train the model. This runs the main.py script.

### ***Test the Model***
Upload an image of a Moringa leaf to test its classification as Healthy or Diseased.

### ***Evaluate the Model***
Evaluate the model's performance on a test dataset.
View the generated confusion matrix and classification metrics.

## **Output Examples**

### ***Confusion Matrix***
Visualizes how well the model performs across different classes.Confusion Matrix

### ***Classification Report Metrics***
Precision, Recall, and F1-Score for each class.Classification Report

## **Technical Details**

### ***Deep Learning Model***
The project utilizes a convolutional neural network (CNN) based on the Inception V4 architecture:

Input size: 255x255
Classes: 2 (Healthy, Diseased)
Optimizer: Adam
Loss Function: Binary Crossentropy

### ***Technologies Used***
Frontend: HTML, Bootstrap, CSS
Backend: Flask
Model Framework: TensorFlow/Keras
Data Visualization: Matplotlib, Seaborn

### ***Customization***
Modify Model Architecture: Edit the main.py script to adjust the CNN layers, learning rates, or hyperparameters.
Add More Classes: Update the dataset and reconfigure the code to handle multiple classes.
Styling: Edit the styles.css file for custom styling.

## **Contribution**
Contributions are welcome! Feel free to:

Fork the repository
Create a new branch
Submit a pull request

## **License**
This project is licensed under the MIT License. You are free to use, modify, and distribute this software.

## **FAQs**
Q1. What image formats are supported for testing?
A: The application supports common image formats like JPG, PNG, and JPEG.

Q2. What should be the size of uploaded images?
A: The images will be resized to 255x255 during preprocessing, so the size of the uploaded image does not matter.

Q3. How can I improve the model's performance?
A: Refer to the suggestions in the README.md under "Customization," where you can modify the architecture, add data augmentation, or fine-tune hyperparameters.

## **Known Issues**
Large Model Training Times:
If the training process takes too long, consider reducing the dataset size for testing or using a pre-trained model with transfer learning.

Memory Issues with Large Datasets:
If memory is insufficient during evaluation, batch processing and optimization techniques like using a generator should help.

## **Future Enhancements**
Add Multi-Class Classification:
Extend the model to classify more types of diseases or plant conditions.

Deploy on Cloud Platforms:
Deploy the app on platforms like AWS, Azure, or Google Cloud for accessibility.

Mobile Compatibility:
Build a mobile-friendly version of the app.

Real-Time Predictions:
Implement real-time leaf classification using camera input.

Localization:
Add multi-language support for broader usability.



### ***Contact***
For questions or suggestions, please contact:

Name: Md Nahijul Islam Niloy
Email: nniloy888@gmail.com
GitHub: [Nahijul Islam Niloy](https://github.com/c-onfused69)
Happy Coding! 😊