# AI Plant Disease Detection Model
# This single-file script is fully automated and optimized for a balance of
# speed and accuracy in a standard Google Colab environment (12GB RAM).

# --- 1. Setup and Imports ---
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Rescaling
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt

# --- GPU Check ---
# This is critical for performance. Training without a GPU will be extremely slow.
print("--- Checking for GPU ---")
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("WARNING: No GPU detected. Training will be very slow.")
    print("In Google Colab, go to 'Runtime' -> 'Change runtime type' and select 'GPU' as the hardware accelerator.")
else:
    # Get the name of the GPU
    gpu_name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'Unknown')
    print(f"SUCCESS: GPU detected: {gpu_name}")
print("----------------------\n")


# This helper is specifically for Google Colab to allow easy file uploads.
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# --- 2. Data Loading and Preparation (Memory Optimized) ---

def load_and_prepare_dataset():
    """
    Loads the PlantVillage dataset using TensorFlow Datasets, optimizes it for memory,
    and prepares it for training.
    """
    print("Loading PlantVillage dataset from TensorFlow Datasets...")
    
    # SPEED BOOST: Using a 50% slice of the data (40% train, 10% val)
    # for a much faster training time while maintaining good accuracy.
    (ds_train, ds_val), ds_info = tfds.load(
        'plant_village',
        split=['train[:40%]', 'train[40%:50%]'],
        shuffle_files=True,
        as_supervised=True, # Returns (image, label) tuples
        with_info=True,
    )

    num_classes = ds_info.features['label'].num_classes
    class_names = ds_info.features['label'].names
    print(f"Dataset loaded. Using a subset for efficiency. Found {num_classes} classes.")

    # --- Data Pipeline Configuration ---
    IMG_SIZE = 128
    BATCH_SIZE = 32 # A balanced batch size for Colab's memory

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])

    def preprocess_image(image, label):
        """Resizes and normalizes images."""
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image, label

    # Build the efficient data pipelines
    ds_train = ds_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(1000) # Shuffle the dataset
    ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(BATCH_SIZE)
    # IMPORTANT: We DO NOT use .cache() here to conserve RAM.
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(BATCH_SIZE)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, num_classes, class_names

# --- 3. Build the Efficient CNN Model ---

def build_efficient_model(num_classes):
    """
    Builds and compiles a CNN model that is a balance of accuracy and efficiency.
    """
    model = Sequential([
        # Input Layer - includes resizing and normalization
        layers.InputLayer(input_shape=(128, 128, 3)),
        Rescaling(1./255),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Fully Connected Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') # Output layer
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

# --- 4. Train the Model ---

def train_model(model, ds_train, ds_val, epochs=20):
    """
    Trains the model with callbacks for efficiency and returns the history.
    """
    # Callbacks to make training more efficient
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2, min_lr=0.00001)

    print("\nStarting model training...")
    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_val,
        callbacks=[early_stopping, reduce_lr]
    )
    print("Training finished.")
    return history

# --- 5. Evaluate and Plot ---

def plot_training_history(history):
    """Plots the training and validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    plt.show()

# --- 6. Prediction on User-Uploaded Images ---

def predict_uploaded_image(model, class_names):
    """
    Handles file upload in Colab, processes the image, and predicts its class.
    """
    if not IN_COLAB:
        print("\nImage upload is only available in Google Colab.")
        print("Please provide a local file path.")
        while True:
            try:
                img_path = input("Enter the path of the image you want to predict (or type 'exit' to quit): ")
                if img_path.lower() == 'exit': break
                if os.path.exists(img_path):
                    predict_single_image(model, class_names, img_path)
                else:
                    print("Error: File not found.")
            except Exception as e:
                print(f"An error occurred: {e}")
        return

    # Colab-specific upload logic
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"\nProcessing '{filename}'...")
        predict_single_image(model, class_names, filename)

def predict_single_image(model, class_names, img_path):
    """Loads a single image, predicts, and displays the result."""
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch

    # Model does not have a rescaling layer, so we do it here.
    # Note: Our efficient model *does* have a Rescaling layer, so this line is not needed.
    # img_array = img_array / 255.0 

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    print(f"  - Prediction: {predicted_class_name}")
    print(f"  - Confidence: {confidence:.2f}%")

    img_display = load_img(img_path)
    plt.imshow(img_display)
    plt.title(f"Predicted: {predicted_class_name} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# --- Main Execution Block ---

if __name__ == '__main__':
    # Step 1: Load and prepare the dataset
    ds_train, ds_val, num_classes, class_names = load_and_prepare_dataset()

    # Step 2: Build the model
    model = build_efficient_model(num_classes)

    # Step 3: Train the model
    history = train_model(model, ds_train, ds_val)

    # Step 4: Save the final model
    model.save("plant_disease_model_efficient.h5")
    print("\nModel saved as plant_disease_model_efficient.h5")

    # Step 5: Evaluate the model's final accuracy
    print("\nEvaluating final model accuracy on the validation dataset...")
    loss, accuracy = model.evaluate(ds_val)
    print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Validation Loss: {loss:.4f}")

    # Step 6: Plot results
    plot_training_history(history)

    # Step 7: Enter prediction loop
    print("\n\n--- Ready for Prediction ---")
    while True:
        try:
            predict_uploaded_image(model, class_names)
            another = input("Predict another image? (y/n): ")
            if another.lower() != 'y':
                break
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            break
