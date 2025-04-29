#MobileNetV2 model

import numpy as np
import time
import os
import psutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, Callback  # type: ignore
from tensorflow.keras.models import load_model # type: ignore

def preprocess_images():
  X = np.load("image_numpys/X_rgb_size50_allImgs.npy")  # Image data
  y = np.load("image_numpys/y_rgb_size50_allImgs.npy")  # Labels

  # Load the preprocessed dataset
  print("Shape of X:", X.shape)  # (num_samples, height, width, channels)
  print("Shape of y:", y.shape)  # (num_samples,)

  # Example: Display first image and label
  #plt.imshow(X[0].reshape(128, 128), cmap="gray")  # Adjust size if needed
  #plt.title(f"Label: {y[0]}")
  #plt.show()

  label_encoder = LabelEncoder()
  y_encoded = label_encoder.fit_transform(y)
  n_classes = len(np.unique(y_encoded))
  y_onehot = to_categorical(y_encoded, num_classes = n_classes)

  return X, y_onehot, n_classes, label_encoder


class StopTrainingAtAccuracy(Callback):
  def __init__(self, target_accuracy):
    super().__init__()
    self.target_accuracy = target_accuracy

  def on_epoch_end(self, epoch, logs=None):
    val_accuracy = logs.get('val_accuracy')
    if val_accuracy and val_accuracy >= self.target_accuracy:
      print(f"\nStopping training as validation accuracy reached {val_accuracy:.2f} (>= {self.target_accuracy})")
      self.model.stop_training = True


def mobilenet(img_size, epochs=50):
  X, y, n_classes, label_encoder = preprocess_images()

  X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)  #train size = 60%
  X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42) #val size = 40% * 0.5 = 20%, test size = 40% * 0.5 = 20%

  base_model = MobileNetV2(weights='imagenet', include_top = False, input_shape = (img_size, img_size, 3))

  #freeze base layers
  for layer in base_model.layers:
    layer.trainable = False

  #add layers to mobilenet architecture
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(128, activation = 'relu')(x)
  x = Dropout(0.5)(x)
  output = Dense(n_classes, activation = 'softmax')(x)

  model = Model(inputs = base_model.input, outputs = output)
  model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
  early_stop = EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True)
  stop_at_accuracy = StopTrainingAtAccuracy(target_accuracy=0.99)

  start_train = time.time()
  history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = epochs,
    batch_size = 16,
    callbacks = [early_stop, stop_at_accuracy],
    verbose = 2
  )
  end_train = time.time()
  train_duration = end_train - start_train
  print(f"Training Time: {train_duration:.2f} seconds")

  val_loss, val_accuracy = model.evaluate(X_val, y_val)
  test_loss, test_accuracy = model.evaluate(X_test, y_test)
  print(f'Val accuracy: {val_accuracy*100:.2f}%, Test accuracy: {test_accuracy*100:.2f}%')

  model.save("mobileNetV2.keras")
  return model


def measure_inference_time(model_path, test_images):  #test_images is numpy array
  model = load_model(model_path)

  # Measure inference time
  start_time = time.time()
  predictions = model.predict(test_images, verbose=0)
  end_time = time.time()

  # Calculate average inference time
  total_time = end_time - start_time
  avg_time = total_time / len(test_images)
  print(f"Average inference time per image: {avg_time:.6f} seconds")

  return predictions


def inference_time_test():
  X = np.load("image_numpys/X_rgb_size50_allImgs.npy")  
  test_images = X[:10]  
  predictions = measure_inference_time("mobileNetV2.keras", test_images)


def test():
  model = mobilenet(img_size=50, epochs=50)


inference_time_test()

