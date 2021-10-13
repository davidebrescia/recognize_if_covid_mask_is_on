
# TEST IMAGES: WIDTH AND HEIGHT
#-------------------------------------------------------------------------------


# This script shows an experiment we made to see which was the most common image size for
# both the training and testing set.
# We are aware of the fact that an Deep Neural Network has to be built to deal with all type
# of image size, but we noticed that with this trick we had less distorted images and faces
# that could bring us to better performances.

import statistics

# test and training folder
training_dir = os.path.join(dataset_dir, 'training')
test_dir = os.path.join(dataset_dir, 'test')

# Next method for iterating in the folders
image_filenames_train = next(os.walk(training_dir))[2]

image_filenames_test = next(os.walk(test_dir))[2]

# 2 Lists where to save the width and height of the train images
width_train = []
height_train = []

# Train images
for image_filename in image_filenames_train:
  img = Image.open(os.path.join(training_dir,image_filename)).convert('RGB')
  w, h = img.size

  # we append into that list the width and height of the image of this iteration
  width_train.append(w)
  height_train.append(h)

# Finally we printed the mode of them to see which is the most common one
print(statistics.mode(width))
print(statistics.mode(height))

# 2 Lists where to save the width and height of the test images
width_test = []
height_test = []



# Test images
for image_filename in image_filenames_test:
  img = Image.open(os.path.join(test_dir,image_filename)).convert('RGB')
  w, h = img.size

  # we append into that list the width and height of the image of this iteration
  width_test.append(w)
  height_test.append(h)

# Finally we printed the mode of them to see which is the most common one
print(statistics.mode(width))
print(statistics.mode(height))