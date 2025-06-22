#/usr/bin/python3
from pathlib import Path
import numpy as np
import torch

TEST_IMAGES_PATH = '../data/t10k-images-idx3-ubyte'
TEST_LABELS_PATH = '../data/t10k-labels-idx1-ubyte'
TEST_IMAGES_OUTPUT_PATH = '../data/test/images'
TEST_LABELS_OUTPUT_PATH = '../data/test/labels'
TEST_COUNT = 100

TRAIN_IMAGES_PATH = '../data/train-images-idx3-ubyte'
TRAIN_LABELS_PATH = '../data/train-labels-idx1-ubyte'
TRAIN_OUTPUT_PATH = '../data/weights.h'

# Magic bytes of MNIST files.
IMAGE_MAGIC = b'\x00\x00\x08\x03'
LABEL_MAGIC = b'\x00\x00\x08\x01'
IMAGE_BEGIN = 16
LABEL_BEGIN = 8

def extract_test():
  # Create output directories.
  Path(TEST_IMAGES_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
  Path(TEST_LABELS_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

  # Open the test images file.
  f = open(TEST_IMAGES_PATH, 'rb')
  data = f.read()

  # Check for magic byte.
  if data[0:4] != IMAGE_MAGIC:
    print('Failed to read test images')
    exit()

  # Get image count and image size.
  image_count    = int.from_bytes(data[4:8],   "big")
  image_row_size = int.from_bytes(data[8:12],  "big")
  image_col_size = int.from_bytes(data[12:16], "big")

  # Print information about the dataset.
  print(f'Image count: {image_count}')
  print(f'Image size:  {image_row_size}x{image_col_size}')

  # Extract images.
  print('Extracting test images into separate text files.')
  # for i in range(image_count):
  for i in range(TEST_COUNT):
    image = []
    for j in range(image_row_size * image_col_size):
      image.append(data[IMAGE_BEGIN + j + i * (image_row_size * image_col_size)])
      
      # Save each image to its own file for easier reading in Verilog-AMS.
      # Each pixel is on a single line.
      with open(f'{TEST_IMAGES_OUTPUT_PATH}/{i}.txt', 'w') as out_file:
          out_file.writelines(f'{value}\n' for value in image)
          
  # Close the test images file.
  f.close()
  
  # Open the test labels file.
  print('Extracting test labels into separate text files.')
  f = open(TEST_LABELS_PATH, 'rb')
  data = f.read()

  # Check for magic byte.
  if data[0:4] != LABEL_MAGIC:
    print('Failed to read test labels')
    exit()

  # Extract labels.
  print('Extracting test labels into separate text files.')
  # for i in range(image_count):
  
  for i in range(TEST_COUNT):
    with open(f'{TEST_LABELS_OUTPUT_PATH}/{i}.txt', 'w') as out_file:
      out_file.writelines(f'{data[LABEL_BEGIN + i]}\n')

def train():
  # Open the train files.
  train_images_file = open(TRAIN_IMAGES_PATH, 'rb')
  train_images_data = train_images_file.read()

  train_labels_file = open(TRAIN_LABELS_PATH, 'rb')
  train_labels_data = train_labels_file.read()

  # Check for magic bytes.
  if train_images_data[0:4] != IMAGE_MAGIC:
    print('Failed to read train images')
    exit()

  if train_labels_data[0:4] != LABEL_MAGIC:
    print('Failed to read train labels')
    exit()

  # Get image count and image size.
  image_count    = int.from_bytes(train_images_data[4:8],   "big")
  image_row_size = int.from_bytes(train_images_data[8:12],  "big")
  image_col_size = int.from_bytes(train_images_data[12:16], "big")

  # Create pytorch model.
  model = torch.nn.Linear(image_row_size * image_col_size, 10)
  loss_func = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

  # Load all images and labels
  images = np.zeros((image_count, image_row_size * image_col_size), dtype=np.uint8)
  labels = np.zeros((image_count,), dtype=np.uint8)

  for i in range(image_count):
    for j in range(image_row_size * image_col_size):
      images[i][j] = train_images_data[IMAGE_BEGIN + j + i * (image_row_size * image_col_size)]
    labels[i] = train_labels_data[LABEL_BEGIN + i]

  # Start training.
  epoch_count = 50
  batch_size = 128
  batch_count = int(image_count + batch_size - 1) // batch_size

  for epoch in range(epoch_count):
    indices = np.arange(image_count)
    np.random.shuffle(indices)

    for i in range(batch_count):
      batch_start = i * batch_size
      batch_end = min(batch_start + batch_size, image_count)
      batch_indices = indices[batch_start:batch_end]

      optimizer.zero_grad()
      outputs = model(torch.tensor(images[batch_indices], dtype=torch.float32) / 255.0)
      loss = loss_func(outputs, torch.tensor(labels[batch_indices], dtype=torch.long))
      loss.backward()
      optimizer.step()

      if (i % batch_count == 300):
        print(f'Epoch: {epoch}, Batch {i + (epoch) * batch_count}: Loss {loss.item():.4f}')

  # Quantize and export weights to C header file.
  weights = model.weight.detach().numpy()
  biases = model.bias.detach().numpy()

  scale = 2 ** 16
  weights_int32 = (weights * scale).astype(np.int32)
  biases_int32 = (biases * scale).astype(np.int32)

  with open(TRAIN_OUTPUT_PATH, 'w') as f:
    f.write('// Weights for a linear classifier, generated via python script\n')
    f.write(f'#define INPUT_SIZE {weights_int32.shape[1]}\n')
    f.write(f'#define OUTPUT_SIZE {weights_int32.shape[0]}\n')

    f.write('static const int32_t WEIGHTS[OUTPUT_SIZE][INPUT_SIZE] = {\n')
    for row in weights_int32:
      f.write('  { ' +  ', '.join(map(str, row)) + ' },\n')
    f.write('};\n\n')

    f.write('static const int32_t BIASES[OUTPUT_SIZE] = {')
    f.write('  ' + ', '.join(map(str, biases_int32)))
    f.write('};\n')

  # Close files.
  train_images_file.close()
  train_labels_file.close()

if __name__ == "__main__":
  # Read MNIST training files.  
  extract_test()
  train()