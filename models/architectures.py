import tensorflow as tf
from tensorflow.keras.applications import ( # type: ignore
    InceptionV3, ResNet50, DenseNet121, EfficientNetB0, MobileNetV3Large
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore

def get_base_model(model_name, input_shape):
    """
    Returns the appropriate base model architecture pre-trained on ImageNet.
    """
    if model_name.lower() == "inception_v3":
        return InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name.lower() == "resnet50":
        return ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name.lower() == "densenet121":
        return DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name.lower() == "efficientnetb0":
        return EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name.lower() == "mobilenetv3":
        return MobileNetV3Large(weights="imagenet", include_top=False, input_shape=input_shape)
    elif model_name.lower() == "vit":
        try:
            from vit_keras import vit
            # Note: ViT input size is strictly 224x224 or 384x384. Using 224x224 here.
            return vit.vit_b16(
                image_size=input_shape[0],
                activation='sigmoid',
                pretrained=True,
                include_top=False,
                pretrained_top=False
            )
        except ImportError:
            raise ImportError("Please install vit-keras to use the ViT model: pip install vit-keras")
    else:
        raise ValueError(f"Model {model_name} not supported.")

def build_model(model_name, input_shape, num_classes):
    """
    Builds the complete model with the chosen base architecture and a custom classification head.
    """
    base_model = get_base_model(model_name, input_shape)
    
    # Freeze the base model for phase 1 training
    base_model.trainable = False
    
    x = base_model.output
    if len(x.shape) == 4:
        x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    
    # Classification head
    if num_classes == 2:
        predictions = Dense(1, activation="sigmoid")(x)
        loss_fn = "binary_crossentropy"
    else:
        predictions = Dense(num_classes, activation="softmax")(x)
        loss_fn = "sparse_categorical_crossentropy"
        
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model, base_model, loss_fn

def unfreeze_top_layers(base_model, unfreeze_fraction=0.3):
    """
    Unfreezes the top layers of the base model for fine-tuning.
    """
    base_model.trainable = True
    num_layers = len(base_model.layers)
    freeze_until = int(num_layers * (1 - unfreeze_fraction))
    
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
        
    return base_model
