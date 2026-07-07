import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for, flash
import tensorflow as tf
from werkzeug.utils import secure_filename

from config import Config
from explainability.gradcam import get_last_conv_layer_name, make_gradcam_heatmap, save_and_display_gradcam
from explainability.lime_explainer import get_lime_explanation
from skimage.segmentation import mark_boundaries

# Suppress TensorFlow verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
app.secret_key = "research_secret_key"
app.config['UPLOAD_FOLDER'] = Config.UPLOADS_DIR
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load a default model (assumes inception_v3 fold 0 is trained)
DEFAULT_MODEL_PATH = os.path.join(Config.OUTPUTS_DIR, "inception_v3", "fold_0", "best_model.keras")
model = None

try:
    if os.path.exists(DEFAULT_MODEL_PATH):
        model = tf.keras.models.load_model(DEFAULT_MODEL_PATH)
        print("Default model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(image_path, target_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_size)
    return tf.expand_dims(image, 0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def train_model():
    return render_template("train.html")

@app.route("/test")
def test_model():
    return render_template("test.html", model_loaded=(model is not None))

@app.route("/evaluate")
def evaluate_model():
    metrics = None
    metrics_path = os.path.join(Config.OUTPUTS_DIR, "inception_v3", "fold_0", "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    return render_template("evaluate.html", metrics=metrics)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("Model not loaded. Please train the model first.", "danger")
        return redirect(url_for("test_model"))
        
    if "image" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("test_model"))
        
    file = request.files["image"]
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(url_for("test_model"))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict
            img_tensor = preprocess_image(filepath, Config.INPUT_SHAPE_INCEPTION[:2])
            preds = model.predict(img_tensor, verbose=0)
            
            if preds.shape[1] == 1:
                prob = preds[0][0]
                pred_class_idx = 1 if prob > 0.5 else 0
                confidence = prob if pred_class_idx == 1 else 1 - prob
            else:
                pred_class_idx = np.argmax(preds[0])
                confidence = preds[0][pred_class_idx]
                
            predicted_class = Config.CLASS_NAMES[pred_class_idx]
            
            # Generate Grad-CAM
            last_conv_name, nested_name = get_last_conv_layer_name(model)
            heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_name, nested_model_name=nested_name)
            
            cam_filename = f"cam_{filename}"
            cam_path = os.path.join(Config.STATIC_DIR, cam_filename)
            save_and_display_gradcam(filepath, heatmap, cam_path)
            
            # Generate LIME
            explanation = get_lime_explanation(model, img_tensor.numpy()[0], num_samples=100) # fast lime for web
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            lime_img = mark_boundaries(temp, mask)
            lime_img = np.uint8(255 * lime_img)
            
            lime_filename = f"lime_{filename}"
            lime_path = os.path.join(Config.STATIC_DIR, lime_filename)
            cv2.imwrite(lime_path, cv2.cvtColor(lime_img, cv2.COLOR_RGB2BGR))
            
            return render_template(
                "test.html", 
                model_loaded=True,
                prediction=predicted_class,
                confidence=f"{confidence*100:.2f}%",
                original_img=url_for('static', filename=f'../uploads/{filename}'),
                cam_img=url_for('static', filename=cam_filename),
                lime_img=url_for('static', filename=lime_filename)
            )
            
        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")
            return redirect(url_for("test_model"))
            
    else:
        flash("Allowed image types are -> png, jpg, jpeg", "danger")
        return redirect(url_for("test_model"))

# Expose uploads folder for serving images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
