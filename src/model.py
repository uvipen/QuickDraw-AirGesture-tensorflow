"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import tensorflow as tf
from src.utils import CLASS_IDS


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 5, activation='relu', use_bias=True, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 5, activation='relu', use_bias=True),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASS_IDS))
    ])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model
