import os
import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import pytesseract
import shutil
from sklearn.model_selection import train_test_split

# Paths
source_real_dir = '.\\dataset\\Data'  # Folder containing real images in class subfolders
source_synthetic_dir = '.\\dataset\\SyntheticData'  # Folder containing synthetic images
output_base_dir = '.\\dataset\\training'  # Output folder for train, val, and test sets

train_dir = os.path.join(output_base_dir, 'train')
val_dir = os.path.join(output_base_dir, 'val')
test_dir = os.path.join(output_base_dir, 'test')

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Function to split and copy images into train, val, and test sets
def split_and_copy_images(source_dirs, output_base_dir, max_images_per_class=100):
    """
    Split images from source directories into train, val, and test sets.

    Parameters:
    - source_dirs: list of source directories containing images for different classes.
    - output_base_dir: path to the output base directory where train, val, and test sets will be created.
    - max_images_per_class: maximum number of images to copy for each class.
    """

    for source_dir in source_dirs:
        for class_name in os.listdir(source_dir):
            class_path = os.path.join(source_dir, class_name)
            images = os.listdir(class_path)

            # Split images into train, val, and test sets
            train_images, temp_images = train_test_split(images, train_size=train_ratio, random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (val_ratio + test_ratio),
                                                       random_state=42)

            # Copy images to respective folders with a maximum limit
            for subset, subset_images in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
                subset_dir = os.path.join(output_base_dir, subset, class_name)
                os.makedirs(subset_dir, exist_ok=True)
                for image_name in subset_images[:max_images_per_class]:
                    shutil.copy(os.path.join(class_path, image_name), os.path.join(subset_dir, image_name))

# Function to train the model
def train_model(data_type='real'):
    """
    Train a CNN model using either real data only or real + synthetic data.

    Parameters:
    - data_type: 'real' to use only real data, 'combined' to use real + synthetic data.
    """
    # Clear any previous data in the output directories
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)

    # Prepare the dataset based on the selected data type
    if data_type == 'real':
        print("Using real data only...")
        split_and_copy_images([source_real_dir], output_base_dir)
    elif data_type == 'combined':
        print("Using real + synthetic data...")
        split_and_copy_images([source_real_dir, source_synthetic_dir], output_base_dir)

    print("Dataset generation complete.")

    # Define image size and batch size
    IMG_HEIGHT, IMG_WIDTH = 128, 128
    BATCH_SIZE = 32

    # Data augmentation and normalization for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Normalization for validation and test sets
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Load validation data
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Load test data
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Build the CNN model
    def build_model():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')  # 3 output classes (Mild, Moderate, Non-Demented)
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # Set up callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
        ModelCheckpoint(f'alzheimers_cnn_best_model_{data_type}.keras', save_best_only=True)
    ]

    # Train the model
    print(f"Training model with {data_type} data...")
    model = build_model()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss ({data_type} data): {test_loss:.4f}")
    print(f"Test Accuracy ({data_type} data): {test_accuracy:.4f}")

    # Save the final model
    model.save(f'alzheimers_cnn_final_model_{data_type}.keras')

# Train with real data only
train_model(data_type='real')

# Train with combined real + synthetic data
train_model(data_type='combined')
