import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import (
    Sequential,
    layers
)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "digit_model.keras"
)

EPOCHS = 12
BATCH_SIZE = 128


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("     HANDWRITTEN NUMBER RECOGNITION AI")
print("     IMPROVED CNN TRAINING")
print("=" * 70)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MNIST
# ============================================================

print("\n[1/7] Loading MNIST dataset...")

(
    x_train,
    y_train
), (
    x_test,
    y_test
) = mnist.load_data()


print(
    f"Training images : {x_train.shape}"
)

print(
    f"Testing images  : {x_test.shape}"
)


# ============================================================
# PREPROCESS DATA
# ============================================================

print("\n[2/7] Preprocessing images...")


# Convert uint8 to float32

x_train = (
    x_train.astype(
        "float32"
    ) / 255.0
)

x_test = (
    x_test.astype(
        "float32"
    ) / 255.0
)


# Add channel dimension

x_train = np.expand_dims(
    x_train,
    axis=-1
)

x_test = np.expand_dims(
    x_test,
    axis=-1
)


print(
    "Images normalized successfully."
)

print(
    "Input shape:",
    x_train.shape
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

print("\n[3/7] Creating handwriting augmentation...")


augmentation = Sequential(
    [

        # Small rotation
        layers.RandomRotation(
            factor=0.08,
            fill_mode="constant",
            fill_value=0.0
        ),

        # Move handwriting horizontally
        layers.RandomTranslation(
            height=0.10,
            width=0.10,
            fill_mode="constant",
            fill_value=0.0
        ),

        # Slight zoom
        layers.RandomZoom(
            height=0.10,
            width=0.10,
            fill_mode="constant",
            fill_value=0.0
        ),

        # Slight contrast variation
        layers.RandomContrast(
            factor=0.15
        )

    ],
    name="handwriting_augmentation"
)


print(
    "Augmentation enabled:"
)

print(
    "  - Rotation"
)

print(
    "  - Translation"
)

print(
    "  - Zoom"
)

print(
    "  - Contrast variation"
)


# ============================================================
# BUILD CNN
# ============================================================

print("\n[4/7] Building improved CNN model...")


model = Sequential(
    [

        layers.Input(
            shape=(28, 28, 1)
        ),


        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        augmentation,


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 1
        # ----------------------------------------------------

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(
            0.20
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 2
        # ----------------------------------------------------

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(
            0.25
        ),


        # ----------------------------------------------------
        # CONVOLUTION BLOCK 3
        # ----------------------------------------------------

        layers.Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Dropout(
            0.25
        ),


        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        layers.Flatten(),


        layers.Dense(
            128,
            activation="relu"
        ),

        layers.BatchNormalization(),

        layers.Dropout(
            0.40
        ),


        layers.Dense(
            10,
            activation="softmax"
        )

    ],

    name="HandwrittenDigitCNN"
)


# ============================================================
# COMPILE MODEL
# ============================================================

print("\n[5/7] Compiling model...")


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


# ============================================================
# SHOW MODEL
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=3,

    mode="max",

    restore_best_weights=True,

    verbose=1

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=1,

    min_lr=0.00001,

    verbose=1

)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n[6/7] Training improved CNN...")

print(
    "\nTraining may take a few minutes."
)

print(
    "Please do not close this terminal.\n"
)


history = model.fit(

    x_train,

    y_train,

    validation_split=0.10,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1

)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n[7/7] Evaluating improved model...")


test_loss, test_accuracy = (
    model.evaluate(
        x_test,
        y_test,
        verbose=0
    )
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("                 MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

print(
    f"Best Val Acc  : "
    f"{max(history.history['val_accuracy']) * 100:.2f}%"
)

print(
    f"Epochs Used   : {len(history.history['loss'])}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print("\nModel saved successfully!")

print(
    f"Model location: {MODEL_PATH}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("        IMPROVED TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)