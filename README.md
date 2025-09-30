# Illegal Mining Detection with CNN

This project implements a Convolutional Neural Network (CNN) for detecting illegal mining activities from satellite or aerial imagery. The system includes data preprocessing, model training, and visualization components to display detection results graphically.

## Project Structure
- `data_loader.py`: Module for loading and preprocessing image data
- `model.py`: CNN architecture for illegal mining detection
- `train.py`: Script for training and evaluating the model
- `visualize.py`: Visualization utilities for displaying results
- `predict.py`: Inference functionality for making predictions
- `main.py`: Main application script to run the entire pipeline

## Setup
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Prepare your dataset:
- Place training images in `data/train/`
- Place validation images in `data/val/`
- Place test images in `data/test/`

3. Run the application:
```
python main.py
```

## Features
- CNN-based detection of illegal mining sites
- Performance metrics visualization
- Prediction visualization with heatmaps
- Model evaluation graphs