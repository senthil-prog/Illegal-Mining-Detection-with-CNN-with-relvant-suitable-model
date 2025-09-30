import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from model import MiningDetectionModel
from train import train_model
from predict import MiningDetector
from visualize import Visualizer

def main():
    """
    Main function to run the illegal mining detection pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Illegal Mining Detection with CNN')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'demo'],
                        help='Mode to run: train, predict, or demo')
    parser.add_argument('--model_type', type=str, default='custom', 
                        choices=['custom', 'vgg16', 'resnet50', 'efficientnet'],
                        help='Type of model to use')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to image for prediction (used in predict mode)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data/train/mining', exist_ok=True)
    os.makedirs('data/train/non_mining', exist_ok=True)
    os.makedirs('data/val/mining', exist_ok=True)
    os.makedirs('data/val/non_mining', exist_ok=True)
    os.makedirs('data/test/mining', exist_ok=True)
    os.makedirs('data/test/non_mining', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run in the specified mode
    if args.mode == 'train':
        print("Starting training mode...")
        model, history, evaluation_results = train_model(
            data_dir=args.data_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        print("Training completed successfully!")
        
    elif args.mode == 'predict':
        if args.image_path is None:
            print("Error: --image_path must be provided in predict mode")
            return
        
        print(f"Making prediction on image: {args.image_path}")
        detector = MiningDetector()
        prediction = detector.predict_single_image(args.image_path)
        print(f"Prediction: {'Mining' if prediction > 0.5 else 'Non-Mining'} (Confidence: {prediction:.2f})")
        
    elif args.mode == 'demo':
        print("Running demo mode with synthetic data...")
        # Create synthetic data for demonstration
        create_demo_data()
        
        # Train a small model on synthetic data
        model, history, _ = train_model(
            data_dir='demo_data',
            model_type=args.model_type,
            epochs=5,  # Use fewer epochs for demo
            batch_size=args.batch_size
        )
        
        # Make predictions on test images
        detector = MiningDetector(model_path='results/best_model.h5')
        
        # Get a list of test images
        test_images = []
        for root, _, files in os.walk('demo_data/test'):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(root, file))
        
        if test_images:
            detector.predict_batch(test_images[:5])
        
        print("Demo completed successfully!")

def create_demo_data():
    """
    Create synthetic data for demonstration purposes.
    """
    # Create directories
    for split in ['train', 'val', 'test']:
        for cls in ['mining', 'non_mining']:
            os.makedirs(f'demo_data/{split}/{cls}', exist_ok=True)
    
    # Generate synthetic images (random noise with different patterns)
    np.random.seed(42)
    
    # Number of images per class and split
    counts = {
        'train': 100,
        'val': 20,
        'test': 10
    }
    
    for split, count in counts.items():
        # Mining images (with more red channel for demonstration)
        for i in range(count):
            # Create a pattern that might represent mining (more red)
            img = np.random.rand(224, 224, 3) * 0.5
            img[:, :, 0] += 0.5  # Add more red
            
            # Add some structures that might represent mining operations
            for _ in range(5):
                x, y = np.random.randint(0, 224, 2)
                size = np.random.randint(10, 50)
                img[max(0, x-size//2):min(224, x+size//2), 
                    max(0, y-size//2):min(224, y+size//2), 0] = 1.0
            
            plt.imsave(f'demo_data/{split}/mining/mining_{i}.png', img)
        
        # Non-mining images (more balanced colors)
        for i in range(count):
            # Create a pattern that might represent non-mining (more green/natural)
            img = np.random.rand(224, 224, 3) * 0.5
            img[:, :, 1] += 0.3  # Add more green
            
            plt.imsave(f'demo_data/{split}/non_mining/non_mining_{i}.png', img)
    
    print(f"Created synthetic dataset with {sum(counts.values())*2} images")

if __name__ == "__main__":
    main()