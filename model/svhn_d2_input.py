import tensorflow as tf
from PIL import Image
import numpy as np
from os import path
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

IMAGE_SIZE = 32
L_NUM_CLASSES = 2
D1_NUM_CLASSES = 10
D2_NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 400 
NUM_EXAMPLES_PER_EPOCH_POR_EVAL = 200

def read_labeled_image_list(label_file, img_dir):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(label_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(path.join(img_dir, filename))
        labels.append(int(label))
    return filenames, labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    len_label = input_queue[1]
    d1_label = input_queue[2]
    d2_label = input_queue[3]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, len_label, d1_label, d2_label

def inputs(label_file, img_dir, batch_size):
    """Construct input for evaluation using the Reader ops.

    Args:
      data_dir: Path to the data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      d1_labels: Labels for the first digit. 2D tensor of [batch_size, num_d1_labels] size.
      d2_labels: Labels for the second digit. 2D tensor of [batch_size, num_d2_labels] size.
    """
    # Reads pfathes of images together with their labels
    image_list, label_list = read_labeled_image_list(label_file, img_dir)
    
    len_labels = [ 1 if label_list[i] >= 10 else 0 for i in xrange(0, len(label_list)) ]
    d1_labels = [ label_list[i]/10 if label_list[i] >= 10 else label_list[i]%10 for i in xrange(0, len(label_list)) ]
    d2_labels = [ label_list[i]%10 if label_list[i] >= 10 else label_list[i] for i in xrange(0, len(label_list))]

    images = ops.convert_to_tensor(image_list, dtype=dtypes.string)
    len_labels = ops.convert_to_tensor(len_labels, dtype=dtypes.int32)
    d1_labels = ops.convert_to_tensor(d1_labels, dtype=dtypes.int32)
    d2_labels = ops.convert_to_tensor(d2_labels, dtype=dtypes.int32)    
    
    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, len_labels, d1_labels, d2_labels],
                                                shuffle=False)
    image, len_label, d1_label, d2_label = read_images_from_disk(input_queue)
   
    image.set_shape([32,32,3])
    len_label.set_shape([])
    d1_label.set_shape([])  
    d2_label.set_shape([])

    image_batch, len_label_batch, d1_label_batch, d2_label_batch = tf.train.batch([image, len_label, d1_label, d2_label],
                                                                 batch_size=batch_size)
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, len_label_batch, d1_label_batch, d2_label_batch
