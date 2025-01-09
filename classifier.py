import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

# Project: Fashion MNIST Classification with a Convolutional Neural Network

# --- Data Loading and Preprocessing ---

def load_and_preprocess_data(validation_split=0.2):
    """
    Loads the Fashion MNIST dataset, normalizes pixel values, and splits
    the data into training, validation, and test sets.

    Args:
        validation_split: The proportion of the training data to use for validation.

    Returns:
        A tuple containing:
            - input_x_train: Training data (images).
            - output_y_train: Training labels (one-hot encoded).
            - input_x_val: Validation data (images).
            - output_y_val: Validation labels (one-hot encoded).
            - input_x_test: Test data (images).
            - output_y_test: Test labels (one-hot encoded).
    """
    (input_x_train_full, output_y_train_full), (input_x_test, output_y_test) = fashion_mnist.load_data()

    # Reshape and normalize
    input_x_train_full = input_x_train_full.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    input_x_test = input_x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # One-hot encode labels
    output_y_train_full = to_categorical(output_y_train_full, num_classes=10)
    output_y_test = to_categorical(output_y_test, num_classes=10)

    # Split into training and validation sets
    input_x_train, input_x_val, output_y_train, output_y_val = train_test_split(
        input_x_train_full, output_y_train_full, test_size=validation_split, random_state=42
    )

    return input_x_train, output_y_train, input_x_val, output_y_val, input_x_test, output_y_test

# --- Data Visualization ---

def visualize_sample_image(input_x, output_y):
    """
    Displays a random sample image from the dataset along with its label.

    Args:
        input_x: The input image data.
        output_y: The corresponding labels.
    """
    random_index = np.random.randint(0, input_x.shape[0])
    plt.imshow(input_x[random_index].reshape(28, 28), cmap='gray')
    plt.title(f"Sample Image (Label: {np.argmax(output_y[random_index])})")
    plt.show()

# --- Model Definition ---

def create_cnn_model(input_shape=(28, 28, 1), l2_strength=0.0001, dropout_rate=0.5):
    """
    Creates a Convolutional Neural Network (CNN) model for image classification.

    Args:
        input_shape: The shape of the input images.
        l2_strength: The L2 regularization strength.
        dropout_rate: The dropout rate.

    Returns:
        A compiled Keras CNN model.
    """
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", input_shape=input_shape, kernel_regularizer=l2(l2_strength)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(l2_strength)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(l2_strength)))
    model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(10, activation="softmax"))  # 10 classes for Fashion MNIST

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

    return model

# --- Model Training ---

def train_model(model, input_x_train, output_y_train, input_x_val, output_y_val, epochs=25, batch_size=128):
    """
    Trains the given CNN model.

    Args:
        model: The Keras model to train.
        input_x_train: Training data (images).
        output_y_train: Training labels (one-hot encoded).
        input_x_val: Validation data (images).
        output_y_val: Validation labels (one-hot encoded).
        epochs: The number of training epochs.
        batch_size: The batch size.

    Returns:
        The training history.
    """
    history = model.fit(
        input_x_train, output_y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(input_x_val, output_y_val)
    )
    return history

# --- Model Evaluation ---

def evaluate_model(model, input_x_test, output_y_test):
    """
    Evaluates the model on the test set and prints the results.

    Args:
        model: The trained Keras model.
        input_x_test: Test data (images).
        output_y_test: Test labels (one-hot encoded).
    """
    test_loss, test_accuracy = model.evaluate(input_x_test, output_y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

# --- Plotting ---

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss curves.

    Args:
        history: The training history returned by model.fit().
    """
    # Accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # Load and preprocess data
    input_x_train, output_y_train, input_x_val, output_y_val, input_x_test, output_y_test = load_and_preprocess_data()

    # Visualize a sample image
    visualize_sample_image(input_x_train, output_y_train)

    # Create the CNN model
    model = create_cnn_model()
    model.summary()

    # Train the model
    history = train_model(model, input_x_train, output_y_train, input_x_val, output_y_val)

    # Evaluate the model
    evaluate_model(model, input_x_test, output_y_test)

    # Plot training history
    plot_training_history(history)
