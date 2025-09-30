import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0

class MiningDetectionModel:
    def __init__(self, input_shape=(224, 224, 3), model_type='custom'):
        """
        Initialize the CNN model for illegal mining detection.
        
        Args:
            input_shape (tuple): Input shape for the model (height, width, channels)
            model_type (str): Type of model to use ('custom', 'vgg16', 'resnet50', 'efficientnet')
        """
        self.input_shape = input_shape
        self.model_type = model_type
        
    def build_custom_model(self):
        """
        Build a custom CNN model for illegal mining detection.
        
        Returns:
            tensorflow.keras.models.Model: Compiled model
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fully connected layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification (mining vs non-mining)
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                     tf.keras.metrics.AUC()]
        )
        
        return model
    
    def build_transfer_learning_model(self):
        """
        Build a transfer learning model using pre-trained networks.
        
        Returns:
            tensorflow.keras.models.Model: Compiled model
        """
        # Select base model according to model_type
        if self.model_type == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.model_type == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif self.model_type == 'efficientnet':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            return self.build_custom_model()
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom layers on top
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs, outputs)
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                     tf.keras.metrics.AUC()]
        )
        
        return model
    
    def get_model(self):
        """
        Get the appropriate model based on the specified model_type.
        
        Returns:
            tensorflow.keras.models.Model: Compiled model
        """
        if self.model_type == 'custom':
            return self.build_custom_model()
        else:
            return self.build_transfer_learning_model()