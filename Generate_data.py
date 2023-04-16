from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

class DataGenerator:
    """A class for creating image data generators using Keras ImageDataGenerator.

    Attributes:
        train_dir (str): Path to the directory containing the training images.
        test_dir (str): Path to the directory containing the testing images.
        img_height (int): The desired height of the input images.
        img_width (int): The desired width of the input images.
        batch_size (int): The batch size to use for training and testing.

    Methods:
        
    create_train_generator(self):
        Returns the data generator for training images.
        
    create_val_generator(self):
        Returns the data generator for validation images.
        
    create_test_generator(self):
        Returns the data generator for testing images.
    """
    def __init__(self, train_dir, test_dir, img_height=224, img_width=224, batch_size=32):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        
    def create_data_generators(self):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        # create train and test data generators
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='binary')

        val_generator = datagen.flow_from_directory(
                self.train_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='binary',
                subset='validation')

        test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=self.batch_size,
                class_mode='binary')
        
        return train_generator, val_generator, test_generator

    
class ImagePlotter:
    """
    Attributes:
        generator: An instance of `tf.keras.preprocessing.image.ImageDataGenerator` used 
    for generating batches of images.
    Methods:
        show_images(num_images): Displays a batch of `num_images` images 
    and
     their corresponding labels.
    """
    def __init__(self, train_generator):
        self.train_generator = train_generator
        self.class_names = list(train_generator.class_indices.keys())
        
    def plot_images(self):
        # Get a batch of images and their corresponding labels from the generator
        images, labels = next(self.train_generator)

        # Plot the images and their corresponding labels
        fig, axes = plt.subplots(6, 5, figsize=(20, 20))
        axes = axes.ravel()
        for i in np.arange(0, 30):
            axes[i].imshow(images[i])
            axes[i].set_title(self.class_names[np.argmax(labels[i])], color='r')
            axes[i].axis('off')

        plt.subplots_adjust(wspace=0.01)
        plt.show()

        

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

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