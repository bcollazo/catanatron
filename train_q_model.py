"""
Train Q-Learning model for Catanatron

This script trains a simple Q-learning neural network that predicts Q-values
for state-action pairs.
"""
import os
import glob
import gzip
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration
TRAINING_DATA_DIR = "training_data_2p"  # 2-player training data
MODEL_OUTPUT_DIR = "trained_models"
MODEL_NAME = "q_model_2p_v1"  # 2-player model
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001

# Q-learning expects state features + action encoding as input
# From the codebase, samples.csv has 1002 columns (state features)
# and actions.csv has 2 columns (action encoding)
# So input dimension is 1002 + 2 = 1004
# Output is single Q-value (scalar)

def load_training_data():
    """Load and combine training data from CSV files."""
    print("Loading training data...")

    # Load the main data file (contains samples + actions + rewards combined)
    main_file = os.path.join(TRAINING_DATA_DIR, "main.csv.gz")

    if not os.path.exists(main_file):
        raise FileNotFoundError(f"No training data found at {main_file}")

    print(f"Loading {main_file}...")
    with gzip.open(main_file, 'rt') as f:
        data = pd.read_csv(f)

    print(f"Total samples loaded: {len(data)}")

    return data


def prepare_data(data):
    """Prepare X and y from the main dataframe."""
    print("Preparing training data...")

    # The main.csv contains:
    # - State features (from samples.csv)
    # - Action encoding (from actions.csv)
    # - Reward labels (from rewards.csv)

    # We'll use DISCOUNTED_RETURN as our reward/Q-value label
    # Input: state features + action encoding
    # Output: DISCOUNTED_RETURN (Q-value)

    # Find column indices (actual reward columns in the data)
    reward_cols = ['RETURN', 'TOURNAMENT_RETURN', 'VICTORY_POINTS_RETURN',
                   'DISCOUNTED_RETURN', 'DISCOUNTED_TOURNAMENT_RETURN',
                   'DISCOUNTED_VICTORY_POINTS_RETURN']

    # Get features (everything except rewards)
    feature_cols = [col for col in data.columns if col not in reward_cols]

    X = data[feature_cols].values

    # Use DISCOUNTED_RETURN as target
    if 'DISCOUNTED_RETURN' in data.columns:
        y = data['DISCOUNTED_RETURN'].values
    else:
        # Fallback to RETURN if DISCOUNTED_RETURN not available
        y = data['RETURN'].values

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    return X, y


def build_q_model(input_dim):
    """Build Q-learning neural network."""
    print(f"Building Q-model with input dimension: {input_dim}")

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Single Q-value output
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    return model


def main():
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)

    # Load data
    data = load_training_data()

    # Prepare training data
    X, y = prepare_data(data)

    # Split into train/validation
    split_idx = int(0.9 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Build model
    model = build_q_model(input_dim=X.shape[1])

    print("\nModel summary:")
    model.summary()

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ],
        verbose=1
    )

    # Save model (use .keras extension for Keras 3)
    model_save_path = model_path if model_path.endswith('.keras') else f"{model_path}.keras"
    print(f"\nSaving model to {model_save_path}")
    model.save(model_save_path)

    print("\nTraining complete!")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"\nModel saved to: {model_save_path}")


if __name__ == "__main__":
    main()
