from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

        

class TrainingHistoryPlotter:
    """
    A class to plot the accuracy and loss of a Keras model training history.
    """

    def __init__(self, history):
        """
        Parameters:
        history (History): The history object returned by the `fit` method of a Keras model.
        """
        self.history = history

    def plot(self):
        """
        Plots the accuracy and loss of the model training history.
        """
        # Plot accuracy
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

        ax1.plot(self.history.history['binary_accuracy'],color='#e32b2b')
        ax1.plot(self.history.history['val_binary_accuracy'],color='#ab1313')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Val'], loc='upper left')

        # Plot loss
        ax2.plot(self.history.history['loss'],color='#e32b2b')
        ax2.plot(self.history.history['val_loss'],color='#ab1313')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Val'], loc='upper left')

        # Plot f1_score
        ax3.plot(self.history.history['f1_score'],color='#e32b2b')
        ax3.plot(self.history.history['val_f1_score'],color='#ab1313')
        ax3.set_title('Model f1_score')
        ax3.set_ylabel('f1_score')
        ax3.set_xlabel('Epoch')
        ax3.legend(['Train', 'Val'], loc='upper left')

        # Show the plot
        plt.show()
        
        

class ConfusionMatrix:
    """
    A class that creates and plots a confusion matrix for a trained model.

    Args:
        model: A trained Keras model.
        test_generator: A Keras ImageDataGenerator for the test set.
        class_names: A list of class names in the same order as the output of the model.

    Attributes:
        y_true: An array of true class labels.
        y_pred: An array of predicted class labels.
        cm: A confusion matrix computed using y_true and y_pred.

    Methods:
        plot(): Creates and displays a confusion matrix plot.
    """
    def __init__(self, model, test_generator):
        self.model = model
        self.test_generator = test_generator

    def plot(self, class_names):
        # Get the true labels and predictions
        y_true = self.test_generator.classes
        y_pred = self.model.predict(self.test_generator)

        # Convert the predictions to class labels
        y_pred = np.argmax(y_pred, axis=1)

        # Create the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix as a heatmap with class names as labels
        fig, ax = plt.subplots()
        plt.figure(figsize=(20, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)

        # Set the axis labels and title
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
    
        # Display the plot
        plt.show()