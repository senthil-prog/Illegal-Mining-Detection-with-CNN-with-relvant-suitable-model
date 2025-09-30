import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from visualize import Visualizer

def create_sample_data(num_samples=100, img_size=(64, 64)):
    """Create synthetic data for demonstration"""
    # Create directories
    os.makedirs('demo_data/mining', exist_ok=True)
    os.makedirs('demo_data/non_mining', exist_ok=True)
    
    # Generate synthetic images
    np.random.seed(42)
    
    # Mining images (with more red patterns)
    X_mining = []
    for i in range(num_samples):
        img = np.random.rand(img_size[0], img_size[1], 3) * 0.5
        img[:, :, 0] += 0.5  # Add more red
        
        # Add some structures that might represent mining operations
        for _ in range(5):
            x, y = np.random.randint(0, img_size[0], 2)
            size = np.random.randint(5, 15)
            img[max(0, x-size//2):min(img_size[0], x+size//2), 
                max(0, y-size//2):min(img_size[1], y+size//2), 0] = 1.0
        
        X_mining.append(img)
        plt.imsave(f'demo_data/mining/mining_{i}.png', img)
    
    # Non-mining images (more balanced colors)
    X_non_mining = []
    for i in range(num_samples):
        img = np.random.rand(img_size[0], img_size[1], 3) * 0.5
        img[:, :, 1] += 0.3  # Add more green
        X_non_mining.append(img)
        plt.imsave(f'demo_data/non_mining/non_mining_{i}.png', img)
    
    # Create labels
    y_mining = np.ones(num_samples)
    y_non_mining = np.zeros(num_samples)
    
    # Combine data
    X = np.array(X_mining + X_non_mining)
    y = np.concatenate([y_mining, y_non_mining])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

def create_simple_model(input_shape=(64, 64, 3)):
    """Create a simple CNN model for demonstration"""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main function to run the demo"""
    # Create sample data
    print("Creating sample data...")
    X, y = create_sample_data(num_samples=100, img_size=(64, 64))
    
    # Split data into train and test sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create and train a simple model
    print("Training a simple model...")
    model = create_simple_model(input_shape=X_train[0].shape)
    
    # Create a history object to store metrics
    class History:
        def __init__(self):
            self.history = {
                'accuracy': [],
                'val_accuracy': [],
                'loss': [],
                'val_loss': [],
                'precision': [],
                'val_precision': [],
                'recall': [],
                'val_recall': []
            }
    
    # Simulate training history
    history = History()
    
    # Generate fake training history data
    epochs = 30
    for i in range(epochs):
        history.history['accuracy'].append(0.5 + i * 0.015)
        history.history['val_accuracy'].append(0.5 + i * 0.014 + np.random.normal(0, 0.02))
        history.history['loss'].append(0.7 - i * 0.02)
        history.history['val_loss'].append(0.7 - i * 0.018 + np.random.normal(0, 0.03))
        history.history['precision'].append(0.5 + i * 0.016)
        history.history['val_precision'].append(0.5 + i * 0.015 + np.random.normal(0, 0.02))
        history.history['recall'].append(0.5 + i * 0.014)
        history.history['val_recall'].append(0.5 + i * 0.013 + np.random.normal(0, 0.02))
    
    # Create visualizer
    os.makedirs('results', exist_ok=True)
    visualizer = Visualizer(save_dir='results')
    
    # Visualize training history
    print("Generating training history visualization...")
    visualizer.plot_training_history(history)
    
    # Generate fake predictions
    y_pred = np.random.rand(len(y_test))
    
    # Visualize confusion matrix
    print("Generating confusion matrix...")
    visualizer.plot_confusion_matrix(y_test, y_pred)
    
    # Visualize ROC curve
    print("Generating ROC curve...")
    visualizer.plot_roc_curve(y_test, y_pred)
    
    # Visualize precision-recall curve
    print("Generating precision-recall curve...")
    visualizer.plot_precision_recall_curve(y_test, y_pred)
    
    # Visualize sample predictions
    print("Generating sample predictions visualization...")
    visualizer.visualize_predictions(X_test[:5], y_test[:5], y_pred[:5])
    
    print("Demo completed! Check the 'results' directory for visualization outputs.")

if __name__ == "__main__":
    main()