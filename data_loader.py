import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2

class DataLoader:
    def __init__(self, data_dir='data', img_size=(224, 224), batch_size=32):
        """
        Initialize the data loader for illegal mining detection.
        
        Args:
            data_dir (str): Directory containing the dataset
            img_size (tuple): Target image size for the model
            batch_size (int): Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Create data directories if they don't exist
        os.makedirs(os.path.join(data_dir, 'train', 'mining'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'train', 'non_mining'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', 'mining'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', 'non_mining'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test', 'mining'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test', 'non_mining'), exist_ok=True)
        
        # Data augmentation for training
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation and testing
        self.val_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)
    
    def load_data(self):
        """
        Load and prepare the datasets for training, validation, and testing.
        
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        train_generator = self.train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = self.val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'val'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        test_generator = self.test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def visualize_samples(self, generator, num_samples=5):
        """
        Visualize sample images from a data generator.
        
        Args:
            generator: Data generator to visualize samples from
            num_samples (int): Number of samples to visualize
        """
        images, labels = next(generator)
        plt.figure(figsize=(15, 5))
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(images[i])
            plt.title(f"Class: {'Mining' if labels[i] == 1 else 'Non-Mining'}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model input
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)