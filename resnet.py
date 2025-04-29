#Resnet50 model

import os
import time
import psutil
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import optimizers, callbacks
from keras.applications import ResNet50, EfficientNetB0, VGG16 # type: ignore
from keras.models import Model # type: ignore
from keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Callback class for creating a callback to pass into model.fit().
# The callback saves the following model info to a csv in a folder called train_tracking:
# epoch #, train_time (s), current process ram (MB), train_accuracy, val_accuracy, train_loss, val_loss
class TrainTrackingCallback(callbacks.Callback):
  def __init__(self, save_dir):
    self.save_dir = save_dir
    self.header_written = False
    self.start_time = time.time()


  def on_epoch_end(self, epoch, logs=None):
    epoch_end_time = time.time() 
    train_time = epoch_end_time - self.start_time 

    # get process RAM (MB)
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  
    header = ['epoch', 'train_time (s)', 'ram_usage (MB)', 'train_accuracy (%)', 'val_accuracy (%)', 'train_loss', 'val_loss']

    # write results to csv
    with open(os.path.join(self.save_dir, 'training_log.csv'), 'a', newline='') as f:
      writer = csv.writer(f)
      
      if not self.header_written:
        writer.writerow(header)
        self.header_written = True

      writer.writerow([epoch, train_time, memory_usage, logs['accuracy'], logs['val_accuracy'], logs['loss'], logs['val_loss']])


def preprocess_images():
  X = np.load("image_numpys/X_rgb_all.npy")  # Image data
  y = np.load("image_numpys/y_rgb_all.npy")  # Labels

  # Load the preprocessed dataset
  #print("Shape of X:", X.shape)  # (num_samples, height, width, channels)
  #print("Shape of y:", y.shape)  # (num_samples,)

  # Example: Display first image and label
  #plt.imshow(X[0].reshape(128, 128), cmap="gray")  # Adjust size if needed
  #plt.title(f"Label: {y[0]}")
  #plt.show()

  label_encoder = LabelEncoder()
  y_encoded = label_encoder.fit_transform(y)
  y_categorical = to_categorical(y_encoded)

  return X, y_categorical, label_encoder


def compare_predictions(Y, predictions, nComparisons):  #numpy arrays

  for i in range(nComparisons):
    print(f'y: {Y[i]}  y^:{predictions[i]:.2f}')

  print('\n')


# Method for building a pre-trained model like ResNet50, EfficientNetB0, or VGG16
def build_model(base_model_fn, input_shape, num_classes):
  base_model = base_model_fn(include_top=False, weights="imagenet", input_shape=input_shape)
  #base_model.trainable = False
  for layer in base_model.layers[:-10]:  #freeze all but the last 10 layers
    layer.trainable = False

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.3)(x)
  predictions = Dense(num_classes, activation="softmax")(x)

  model = Model(inputs=base_model.input, outputs=predictions)
  return model


def plot_confusion_matrix(y_true, y_pred_classes, label_encoder):
  conf_matrix = confusion_matrix(y_true, y_pred_classes)

  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()


# Method for training Resnet.
# There will be a delay before training starts.
def resnet(metrics_folder, model_file, epochs=50):
  IMG_SIZE = (128, 128)
  N_CLASSES = 26
  X, y, label_encoder = preprocess_images()  #use label encoder for confusion matrix
  X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)  #train size = 60%
  X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42) #val size = 40% * 0.5 = 20%, test size = 40% * 0.5 = 20%

  datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
  )

  INPUT_SIZE = (128,128,3)
  model = build_model(ResNet50, INPUT_SIZE, N_CLASSES)  #resnet requires RGB input
  #can also use EfficientNetB_ or VGG16

  # Compile the model
  optimizer = optimizers.Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  #early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
 

  checkpoint = callbacks.ModelCheckpoint(
    os.path.join(metrics_folder, 'model_epoch_{epoch:02d}.keras'),
    save_best_only=False,  # Save model at every epoch
    #monitor='val_loss',    # Optional: you can monitor a specific metric like val_loss
    save_weights_only=False,
    verbose=1
  )

  history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[reduce_lr, checkpoint, TrainTrackingCallback(metrics_folder)]
  )

  model.save(os.path.join(metrics_folder, model_file))


  # Evaluate the model
  train_loss, train_acc = model.evaluate(X_train, y_train)
  val_loss, val_acc = model.evaluate(X_val, y_val)
  print(f"Training Accuracy: {train_acc*100:.2f}%")
  print(f"Validation Accuracy: {val_acc*100:.2f}%")


  # Generate predictions
  n_samples = 10
  y_pred = model.predict(X_val)
  #y_val_samples = y_val[:n_samples]
  # compare_predictions(y_val_samples, y_pred, n_samples)
  
  y_pred_classes = np.argmax(y_pred, axis=1)  #returns index of max probability for each prediction
  y_true = np.argmax(y_val, axis=1)
  #compare_predictions(y_true, y_pred_classes, n_samples)
  plot_confusion_matrix(y_true, y_pred_classes, label_encoder)


def test():
  resnet(metrics_folder="model_tracking/resnet_all_tracking", model_file="resnet.keras", epochs=80)


test()