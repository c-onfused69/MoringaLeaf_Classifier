import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import InceptionV3  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping  # type: ignore

# Suppress TensorFlow verbose logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 2 suppresses info; 3 suppresses warnings and errors

# Paths
base_dir = "E:/AAA/MoringaLeaf_Classifier/dataset"
train_dir = os.path.join(base_dir, "traning_set")
validation_dir = os.path.join(base_dir, "validation_set")
test_dir = os.path.join(base_dir, "testing_set")
model_save_path = "E:/AAA/MoringaLeaf_Classifier/models/moringaleaf_model.h5"

# Data Generators with enhanced data augmentation
image_size = (255, 255)  # Updated image size
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    brightness_range=[0.8, 1.2],  # Adjust brightness
    channel_shift_range=30.0,  # Adjust color channels
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
)

# Model Definition using InceptionV3 as the base model
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(255, 255, 3))

# Adding custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tune the model by unfreezing selected layers
for layer in base_model.layers[:250]:  # Adjust the number of trainable layers
    layer.trainable = True
for layer in base_model.layers[250:]:
    layer.trainable = False

# Compile the model with a reduced learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks for learning rate adjustment and early stopping
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
epochs = 50
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    verbose=1,
    callbacks=[lr_reduction, early_stopping]
)

# Save the trained model
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# No automatic testing, but you can manually test the model later
