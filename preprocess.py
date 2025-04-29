import os, shutil
import cv2
import numpy as np

# Function generates numpy files for preprocessed images
def preprocess_images(dir_name, X_output_name, y_output_name, n_images_per_letter, img_size):
    dataset_dir = os.path.join(os.getcwd(), dir_name)
    data = []
    labels = []

    # Loop through each letter folder
    for letter in os.listdir(dataset_dir):
        letter_path = os.path.join(dataset_dir, letter)
        
        if not os.path.isdir(letter_path): continue  # Check if it's a directory
        print(f"Processing images for letter: {letter}")

        # Get the first n images in the folder
        img_names = sorted(os.listdir(letter_path))[:n_images_per_letter]

        for img_name in img_names:
            img_path = os.path.join(letter_path, img_name)

            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

                img = cv2.resize(img, (img_size, img_size))
                img = img / 255.0  # Normalize pixel values (0 to 1)
                data.append(img)
            labels.append(letter)


    # Convert lists to NumPy arrays
    data = np.array(data, dtype=np.float32).reshape(-1, img_size, img_size, 1)  #1 channel for grey imgs, 3 channels for RGB
    labels = np.array(labels)
    np.save(X_output_name, data)
    np.save(y_output_name, labels)

    print(f"Preprocessing complete! Processed {len(data)} images.")


# Function generates numpy files for 2 classes of preprocessed images, A and non-A 
# (equal number images from each other letter class)
def preprocess_2_classes(dir_name, X_output_name, y_output_name, n, img_size=200):
  dataset_dir = os.path.join(os.getcwd(), dir_name)
  data = []
  labels = []

  # Separate "A" images and non-"A" images
  a_images = []
  non_a_images = {}

  for letter in os.listdir(dataset_dir):
    letter_path = os.path.join(dataset_dir, letter)

    if not os.path.isdir(letter_path):
      continue  # Skip if not a directory

    img_names = sorted(os.listdir(letter_path))

    if letter == "A":
      a_images = img_names[:n]  # Take the first n images for "A"
    else:
      non_a_images[letter] = img_names  # Store all images for non-"A" letters


  # Process "A" images
  print("Processing images for class: A")
  for img_name in a_images:
    img_path = os.path.join(dataset_dir, "A", img_name)
    img = cv2.imread(img_path)

    if img is not None:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
      #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
      img = cv2.resize(img, (img_size, img_size))
      img = img / 255.0  # Normalize pixel values (0 to 1)
      data.append(img)
      labels.append("A")


  # Process non-"A" images
  print("Processing images for class: non-A")
  non_a_count_per_class = n // len(non_a_images)  # Distribute n images across non-"A" classes

  for letter, img_names in non_a_images.items():
    img_names = img_names[:non_a_count_per_class]  # Take the first n images for this class

    for img_name in img_names:
      img_path = os.path.join(dataset_dir, letter, img_name)
      img = cv2.imread(img_path)

      if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0  # Normalize pixel values (0 to 1)
        data.append(img)
        labels.append("non-A")

  # Convert lists to NumPy arrays
  data = np.array(data, dtype=np.float32).reshape(-1, img_size, img_size, 1)  # 3 channels for RGB
  labels = np.array(labels)
  np.save(X_output_name, data)
  np.save(y_output_name, labels)

  print(f"Preprocessing complete! Processed {len(data)} images into 2 classes.")


# Function creates a subset folder of the original data that contains n images for each letter class
def create_subset(src_dir, dest_dir, n_imgs_per_class):
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Loop over each folder in the source directory (representing letter classes)
    for letter in os.listdir(src_dir):
        letter_folder = os.path.join(src_dir, letter)
        
        # Check if it's a directory (class folder)
        if os.path.isdir(letter_folder):
            dest_letter_folder = os.path.join(dest_dir, letter)
            
            # Create the letter class folder in the destination directory
            if not os.path.exists(dest_letter_folder):
                os.makedirs(dest_letter_folder)

            images = os.listdir(letter_folder)
            images = sorted(images)[:n_imgs_per_class]  # Take the first n images

            # Copy the selected images to the new folder
            for image in images:
                src_image_path = os.path.join(letter_folder, image)
                dest_image_path = os.path.join(dest_letter_folder, image)
                shutil.copy(src_image_path, dest_image_path)
                #print(f"Copied {image} to {dest_letter_folder}")

    print(f"Subset creation complete! Images copied to {dest_dir}")


# Example calls
src_dir = 'train'  
n_imgs_per_class = 3000

#create_subset(src_dir, dest_dir, n_imgs_per_class)
#preprocess_2_classes(src_dir, 'image_numpys/X_A_grey_size50.npy', 'image_numpys/y_notA_grey_size50.npy', n_imgs_per_class, img_size=50)
#preprocess_images(src_dir, 'image numpys/X_grey_size50_allImgs.npy', 'image numpys/y_grey_size50_allImgs.npy', n_imgs_per_class, img_size=50)