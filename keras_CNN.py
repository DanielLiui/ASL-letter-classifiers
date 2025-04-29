#Fine-tuned CNN model

import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import models, layers, utils, optimizers, callbacks

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, Callback  # type: ignore


def preprocess_images():
  X = np.load("image_numpys/X_grey_size50_allImgs.npy")  # Image data
  y = np.load("image_numpys/y_grey_size50_allImgs.npy")  # Labels

  # Load the preprocessed dataset
  print("Shape of X:", X.shape)  # (num_samples, height, width, channels)
  print("Shape of y:", y.shape)  # (num_samples,)

  # Example: Display first image and label
  plt.imshow(X[0], cmap="gray")  # Adjust size if needed
  plt.title(f"Label: {y[0]}")
  plt.show()

  label_encoder = LabelEncoder()
  y_encoded = label_encoder.fit_transform(y)
  y_categorical = utils.to_categorical(y_encoded)

  return X, y_categorical


class StopTrainingAtAccuracy(Callback):
  def __init__(self, target_accuracy):
    super().__init__()
    self.target_accuracy = target_accuracy

  def on_epoch_end(self, epoch, logs=None):
    val_accuracy = logs.get('val_accuracy')
    if val_accuracy and val_accuracy >= self.target_accuracy:
      print(f"\nStopping training as validation accuracy reached {val_accuracy:.2f} (>= {self.target_accuracy})")
      self.model.stop_training = True


def custom_keras_model(img_size, epochs=50):
  X, y = preprocess_images()
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
  datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
  )
  datagen.fit(X_train)

  model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers. Dropout(0.3),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(y.shape[1], activation='softmax')  # 26 classes
  ])


  # Compile the model
  optimizer = optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
  stop_at_accuracy = StopTrainingAtAccuracy(target_accuracy=0.99)
  reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

  start_train = time.time()
  history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr, stop_at_accuracy]
  )
  end_train = time.time()
  train_duration = end_train - start_train
  print(f"Training Time: {train_duration:.2f} seconds")

  model.save("custom_keras_model.keras")


  # Evaluate the model
  print("\n=== Training Performance ===")
  train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
  print(f"Training Accuracy: {train_acc*100:.2f}%")
  print(f"Training Loss: {train_loss:.4f}")

  print("\n=== Validation Performance ===")
  val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
  print(f"Validation Accuracy: {val_acc*100:.2f}%")
  print(f"Validation Loss: {val_loss:.4f}")


  # Generate predictions
  y_pred = model.predict(X_val)
  y_pred_classes = np.argmax(y_pred, axis=1)
  y_true = np.argmax(y_val, axis=1)

  # Classification report
  # print("\n=== Classification Report ===")
  # print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

  # Confusion matrix (optional - might be large for many classes)
  # print("\n=== Confusion Matrix ===")
  # print(confusion_matrix(y_true, y_pred_classes))


def test():
  custom_keras_model(img_size=50, epochs=50)


test()

  

