import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_loader import DataLoader
from visualize import Visualizer

class MiningDetector:
    def __init__(self, model_path='results/best_model.h5', img_size=(224, 224)):
        """
        Initialize the mining detector with a trained model.
        
        Args:
            model_path (str): Path to the trained model file
            img_size (tuple): Image size for preprocessing
        """
        self.model_path = model_path
        self.img_size = img_size
        self.model = self._load_model()
        self.data_loader = DataLoader(img_size=img_size)
        self.visualizer = Visualizer(save_dir='results')
    
    def _load_model(self):
        """
        Load the trained model.
        
        Returns:
            tensorflow.keras.Model: Loaded model
        """
        if os.path.exists(self.model_path):
            return load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
    
    def predict_single_image(self, image_path, visualize=True):
        """
        Make a prediction on a single image.
        
        Args:
            image_path (str): Path to the image file
            visualize (bool): Whether to visualize the prediction
            
        Returns:
            float: Prediction probability (0-1)
        """
        # Preprocess the image
        img = self.data_loader.preprocess_single_image(image_path)
        
        # Make prediction
        prediction = self.model.predict(img)[0][0]
        
        # Visualize if requested
        if visualize:
            # Load and display the original image
            plt.figure(figsize=(10, 6))
            
            # Original image
            plt.subplot(1, 2, 1)
            img_display = plt.imread(image_path)
            plt.imshow(img_display)
            plt.title(f"Prediction: {'Mining' if prediction > 0.5 else 'Non-Mining'}\nConfidence: {prediction:.2f}")
            plt.axis('off')
            
            # Grad-CAM visualization
            plt.subplot(1, 2, 2)
            gradcam = self.visualizer.generate_gradcam(self.model, img)
            plt.imshow(gradcam)
            plt.title("Regions of Interest (Grad-CAM)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('results/single_prediction.png')
            plt.show()
        
        return prediction
    
    def predict_batch(self, image_paths, visualize=True):
        """
        Make predictions on a batch of images.
        
        Args:
            image_paths (list): List of image file paths
            visualize (bool): Whether to visualize the predictions
            
        Returns:
            numpy.ndarray: Array of prediction probabilities
        """
        # Preprocess images
        preprocessed_images = []
        for path in image_paths:
            img = self.data_loader.preprocess_single_image(path)
            preprocessed_images.append(img[0])  # Remove batch dimension
        
        preprocessed_images = np.array(preprocessed_images)
        
        # Make predictions
        predictions = self.model.predict(preprocessed_images)
        
        # Visualize if requested
        if visualize and len(image_paths) > 0:
            plt.figure(figsize=(15, 5 * min(5, len(image_paths))))
            
            for i, (path, pred) in enumerate(zip(image_paths[:5], predictions[:5])):
                # Original image
                plt.subplot(min(5, len(image_paths)), 2, 2*i+1)
                img_display = plt.imread(path)
                plt.imshow(img_display)
                plt.title(f"Prediction: {'Mining' if pred > 0.5 else 'Non-Mining'}\nConfidence: {pred[0]:.2f}")
                plt.axis('off')
                
                # Grad-CAM visualization
                plt.subplot(min(5, len(image_paths)), 2, 2*i+2)
                img_for_gradcam = np.expand_dims(preprocessed_images[i], axis=0)
                gradcam = self.visualizer.generate_gradcam(self.model, img_for_gradcam)
                plt.imshow(gradcam)
                plt.title("Regions of Interest (Grad-CAM)")
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('results/batch_predictions.png')
            plt.show()
        
        return predictions

if __name__ == "__main__":
    # Example usage
    detector = MiningDetector()
    
    # For demonstration, we'll use a placeholder image path
    # In a real scenario, you would use actual image paths
    image_path = "data/test/mining/sample1.jpg"
    
    # Check if the file exists (this is just for demonstration)
    if os.path.exists(image_path):
        # Predict on a single image
        prediction = detector.predict_single_image(image_path)
        print(f"Prediction: {'Mining' if prediction > 0.5 else 'Non-Mining'} (Confidence: {prediction:.2f})")
    else:
        print(f"Image file not found: {image_path}")
        print("Please place your test images in the appropriate directories and update the paths.")