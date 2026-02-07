"""
model.py - LSTM Caption Generator Model

This module defines the neural network architecture for caption generation.

Architecture:
    Image Features (2048) → Dense (256) → 
    Word Embedding (256) → LSTM (256) → Dense (vocab_size) → Softmax

The model takes:
    1. Video features (from CNN)
    2. Partial caption (what's been generated so far)
    
And predicts:
    Next word in the caption
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def create_caption_model(vocab_size, max_length, embedding_dim=256, lstm_units=256, 
                         feature_dim=2048):
    """
    Create the LSTM-based caption generation model.
    
    Args:
        vocab_size: Size of vocabulary (number of unique words)
        max_length: Maximum caption length
        embedding_dim: Dimension of word embeddings (default: 256)
        lstm_units: Number of LSTM units (default: 256)
        feature_dim: Dimension of image features (default: 2048 for InceptionV3)
    
    Returns:
        Keras Model
    
    Model Overview:
        The model has TWO inputs:
            1. Image features: Visual information about the video
            2. Text sequence: Caption words generated so far
        
        These are combined and passed to LSTM to predict next word.
    
    Example:
        Image: [features from CNN]
        Input caption: "<start> a dog"
        Output prediction: "running" (next word)
    """
    
    # ==================== INPUT 1: IMAGE FEATURES ====================
    # Shape: (None, 2048)
    # These are the features extracted by InceptionV3
    input_image_features = Input(shape=(feature_dim,), name='image_features')
    
    # Project image features to embedding dimension
    # Why? To match the dimension of word embeddings
    # 2048 → 256
    image_dense = Dense(
        embedding_dim, 
        activation='relu',
        kernel_regularizer=l2(0.001),  # Prevent overfitting
        name='image_embedding'
    )(input_image_features)
    
    # Add dropout for regularization
    image_dropout = Dropout(0.3)(image_dense)
    
    
    # ==================== INPUT 2: TEXT SEQUENCE ====================
    # Shape: (None, max_length)
    # This is the partial caption (sequence of word indices)
    input_text_sequence = Input(shape=(max_length,), name='text_sequence')
    
    # Word Embedding Layer
    # Converts word indices to dense vectors
    # Example: word_index 5 → [0.2, 0.5, ..., 0.1] (256 numbers)
    # 
    # What is embedding?
    #   - Learns meaningful representations of words
    #   - Similar words get similar vectors
    #   - Example: "dog" and "puppy" will have similar embeddings
    text_embedding = Embedding(
        input_dim=vocab_size,      # Size of vocabulary
        output_dim=embedding_dim,   # Dimension of embedding vectors
        mask_zero=True,             # Ignore padding (zeros)
        name='text_embedding'
    )(input_text_sequence)
    
    # Add dropout
    text_dropout = Dropout(0.3)(text_embedding)
    
    
    # ==================== COMBINE IMAGE + TEXT ====================
    # We need to combine image features with text sequence
    # Image shape: (None, 256)
    # Text shape: (None, max_length, 256)
    # 
    # Solution: Repeat image features for each time step
    # (None, 256) → (None, 1, 256) → (None, max_length, 256)
    
    from tensorflow.keras.layers import RepeatVector
    
    # Expand image features to match sequence length
    image_repeated = RepeatVector(max_length)(image_dropout)
    
    # Add image and text features element-wise
    # This combines visual and textual information
    merged = Add()([image_repeated, text_dropout])
    
    
    # ==================== LSTM LAYER ====================
    # LSTM processes the sequence and generates caption
    # 
    # What is LSTM?
    #   - Recurrent Neural Network that remembers information
    #   - Processes sequence one word at a time
    #   - Maintains "memory" of what it has seen
    #   - Good for sequential data like text
    # 
    # How it works:
    #   Step 1: Read "<start>" + image → predict "a"
    #   Step 2: Read "a" + remember "<start>" → predict "dog"
    #   Step 3: Read "dog" + remember previous → predict "running"
    #   ...and so on
    
    lstm_out = LSTM(
        lstm_units,
        return_sequences=False,
        name='lstm_layer'
    )(merged)
 
    
    # Add dropout
    lstm_dropout = Dropout(0.3)(lstm_out)
    
    
    # ==================== OUTPUT LAYER ====================
    # Predict probability distribution over vocabulary
    # For each position, predict which word comes next
    # 
    # Output shape: (None, max_length, vocab_size)
    # Each time step gets a probability for each word
    
    output = Dense(
        vocab_size,
        activation='softmax',  # Convert to probabilities
        name='output'
    )(lstm_dropout)
    
    
    # ==================== CREATE MODEL ====================
    model = Model(
        inputs=[input_image_features, input_text_sequence],
        outputs=output,
        name='video_caption_generator'
    )
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
    
    Loss Function:
        - Categorical Crossentropy: Measures how wrong predictions are
        - Lower loss = better predictions
    
    Optimizer:
        - Adam: Adaptive learning rate optimizer
        - Automatically adjusts learning rate during training
    """
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',  # For integer labels
        metrics=['accuracy']  # Track prediction accuracy
    )
    
    print("Model compiled successfully!")
    return model


def model_summary(model):
    """
    Print model architecture and parameter count.
    
    Args:
        model: Keras model
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    print("="*60)
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    print("="*60)


# Example usage and testing
if __name__ == "__main__":
    """
    Test the model creation.
    """
    
    print("Testing model creation...")
    
    # Example parameters
    VOCAB_SIZE = 5000      # Vocabulary size
    MAX_LENGTH = 20        # Maximum caption length
    EMBEDDING_DIM = 256    # Word embedding dimension
    LSTM_UNITS = 256       # LSTM hidden units
    FEATURE_DIM = 2048     # InceptionV3 feature dimension
    
    # Create model
    model = create_caption_model(
        vocab_size=VOCAB_SIZE,
        max_length=MAX_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        lstm_units=LSTM_UNITS,
        feature_dim=FEATURE_DIM
    )
    
    # Compile model
    model = compile_model(model)
    
    # Print summary
    model_summary(model)
    
    # Test with dummy data
    print("\nTesting model with dummy data...")
    import numpy as np
    
    # Create dummy inputs
    dummy_features = np.random.randn(2, FEATURE_DIM)  # 2 videos
    dummy_sequences = np.random.randint(0, VOCAB_SIZE, (2, MAX_LENGTH))
    
    # Forward pass
    predictions = model.predict([dummy_features, dummy_sequences], verbose=0)
    
    print(f"Input features shape: {dummy_features.shape}")
    print(f"Input sequences shape: {dummy_sequences.shape}")
    print(f"Output predictions shape: {predictions.shape}")
    print(f"Expected shape: (2, {MAX_LENGTH}, {VOCAB_SIZE})")
    
    print("\n✓ Model created and tested successfully!")
