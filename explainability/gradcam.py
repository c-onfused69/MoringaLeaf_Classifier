import numpy as np
import tensorflow as tf
import cv2

def get_last_conv_layer_name(model):
    """Finds the last convolutional layer in a model."""
    for layer in reversed(model.layers):
        # Handle nested models (like base_model inside our custom model)
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                try:
                    if len(sub_layer.output.shape) == 4:
                        return sub_layer.name, layer.name
                except Exception:
                    pass
        else:
            try:
                if len(layer.output.shape) == 4:
                    return layer.name, None
            except Exception:
                pass
    raise ValueError("Could not find a convolutional layer.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, nested_model_name=None):
    """
    Generates a Grad-CAM heatmap.
    """
    # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
    if nested_model_name:
        nested_model = model.get_layer(nested_model_name)
        grad_model = tf.keras.models.Model(
            [nested_model.inputs], 
            [nested_model.get_layer(last_conv_layer_name).output, nested_model.output]
        )
        
        # We also need the top classifier part
        classifier_input = tf.keras.Input(shape=nested_model.output.shape[1:])
        x = classifier_input
        for layer in model.layers:
            if layer.name != nested_model_name and not isinstance(layer, tf.keras.layers.InputLayer):
                x = layer(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        
    else:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        classifier_model = None

    with tf.GradientTape() as tape:
        if classifier_model:
            last_conv_layer_output, nested_preds = grad_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(nested_preds)
        else:
            last_conv_layer_output, preds = grad_model(img_array)
            
        if pred_index is None:
            if preds.shape[1] == 1:
                # Binary
                pred_index = 0
            else:
                pred_index = tf.argmax(preds[0])
                
        if preds.shape[1] == 1:
            class_channel = preds[:, 0]
        else:
            class_channel = preds[:, pred_index]

    # Gradient of the output neuron w.r.t. the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Overlays the heatmap on the original image and saves it.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return superimposed_img
