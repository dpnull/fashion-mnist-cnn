# Fashion MNIST CNN Classifier

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The model achieves a test accuracy of approximately 91.44%, demonstrating the effectiveness of CNNs for image recognition tasks.

## Project Description

The Fashion MNIST dataset consists of 60,000 training images and 10,000 test images of 10 different fashion categories. Each image is a 28x28 grayscale image. This project aims to build and train a CNN model that can accurately classify these images into their respective categories.

The model architecture is inspired by the classic LeNet architecture, adapted for the grayscale Fashion MNIST images. It utilizes convolutional layers with ReLU activation for feature extraction, max-pooling layers for dimensionality reduction, and fully connected layers for classification. To prevent overfitting and improve generalization, L2 kernel regularization and dropout are applied.

## Dataset

The Fashion MNIST dataset is used for this project.

*   **Training set:** 60,000 images
*   **Test set:** 10,000 images
*   **Image size:** 28x28 pixels (grayscale)
*   **Number of classes:** 10

**Fashion Categories:**

| Label | Description   |
| ----- | ------------- |
| 0     | T-shirt/top   |
| 1     | Trouser       |
| 2     | Pullover      |
| 3     | Dress         |
| 4     | Coat          |
| 5     | Sandal        |
| 6     | Shirt         |
| 7     | Sneaker       |
| 8     | Bag           |
| 9     | Ankle boot    |

## Results

The CNN model achieved the following performance on the test set after 25 epochs:

*   **Test Loss:** Approximately 0.3551
*   **Test Accuracy:** Approximately 91.44%

The following plots illustrate the training and validation accuracy and loss curves over the 25 epochs:

![Figure_1](https://github.com/user-attachments/assets/4be69582-6f6d-4448-ba8c-6e5019421ce7)

**Analysis of Results:**

*   **Accuracy:** The training accuracy shows a steady increase over the 25 epochs, reaching around 96%. The validation accuracy also improves but plateaus around 92-93% after about 15 epochs. This suggests that the model is learning effectively but might be starting to overfit slightly.
*   **Loss:** The training loss decreases consistently throughout the training process. The validation loss also decreases initially but starts to level off and even slightly increase after around 15 epochs. This further supports the possibility of minor overfitting.
*   **Overfitting:** The slight divergence between training and validation accuracy, along with the plateauing of validation loss, indicates that the model might be starting to memorize the training data and is not generalizing as well to unseen data.
*   **Overall Performance:** Despite the potential for slight overfitting, the model achieves a good test accuracy of 91.44%, demonstrating its ability to effectively classify images from the Fashion MNIST dataset.

**Further Improvements:**

*   **Early Stopping:** Implementing early stopping during training to halt the process when the validation loss stops decreasing, preventing further overfitting.
*   **Hyperparameter Tuning:**  Experimenting with different hyperparameters, such as the learning rate, dropout rate, and L2 regularization strength, to find a better balance between training and validation performance.
*   **Data Augmentation:** Applying data augmentation techniques to increase the diversity of the training data and potentially improve generalization.

## Dependencies

This project requires the following Python libraries:

*   TensorFlow 2.x (`tensorflow`)
*   Keras (`keras`)
*   NumPy (`numpy`)
*   Matplotlib (`matplotlib`)
*   scikit-learn (`sklearn`)

You can install these dependencies using pip:

```bash
pip install tensorflow numpy matplotlib scikit-learn
