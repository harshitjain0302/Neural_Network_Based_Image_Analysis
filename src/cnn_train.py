import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_cnn(input_shape=(64,64,1), num_classes=3):
    model = Sequential([
        Conv2D(16, (3,3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(8, activation="relu"),          # compact representation layer
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_eval(X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]

    model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(set(y_train)))
    start = time.time()
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=batch_size, verbose=1)
    elapsed = time.time() - start

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    return model, hist, {"train_time_sec": elapsed, "val_loss": val_loss, "val_acc": val_acc}
