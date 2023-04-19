import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from colorama import Fore


class Predictor:
    """A class that loads and uses a trained VGG16 model to make predictions on test images.
    Parameters:
        model_path (str): Path to the saved Keras model.
        class_names (list of str): A list of strings representing the class names for the predicted outputs.
        img_dir (str): Path to the directory containing the test images.
    Attributes:
        model (Keras model): The trained Keras model loaded from the given path.
        class_names (list of str): A list of strings representing the class names for the predicted outputs.
        img_dir (str): Path to the directory containing the test images.
        input_data (numpy.ndarray): Numpy array containing the preprocessed input data.
        preds (numpy.ndarray): Numpy array containing the predicted class labels for the test images.
    Methods:
        _load_images: Private method to load and preprocess the test images.
        predict: Public method to make predictions on the loaded test images and print the results.
    """
    def __init__(self, model_path, img_dir, class_names=['Leaf scorch strewberry plant', 'healthy strawberry plant']):
        self.model_path = model_path
        self.class_names = class_names
        self.img_dir = img_dir
        self.model = load_model(model_path)
        self.input_data = None
        self.preds = None
    
    def _load_images(self):
        """Private method to load and preprocess the test images."""
        input_data = []
        for img_path in os.listdir(self.img_dir):
            img = image.load_img(os.path.join(self.img_dir, img_path), target_size=(224, 224))
            img = image.img_to_array(img)
            img = preprocess_input(img)  # add rescaling here
            img = np.expand_dims(img, axis=0)
            input_data.append(img)

        self.input_data = np.concatenate(input_data, axis=0)

    def predict(self):
        """Public method to make predictions on the loaded test images and print the results."""
        self._load_images()
        self.preds = self.model.predict(self.input_data)
        
        for i, img_path in enumerate(os.listdir(self.img_dir)):
            name = self.class_names[int(self.preds[i])]
            print('\n')
            print(Fore.BLACK,f'image {img_path} Classified as :',Fore.RED,name)
