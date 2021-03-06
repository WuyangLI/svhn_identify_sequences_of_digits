import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import svhn_d2_input
from svhn_d2_input import inputs

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 400,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/lwy/Downloads/svhn_data_2_32by32/train',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
# Global constants describing the data set.
IMAGE_SIZE = svhn_d2_input.IMAGE_SIZE
L_NUM_CLASSES = svhn_d2_input.L_NUM_CLASSES
D1_NUM_CLASSES = svhn_d2_input.D1_NUM_CLASSES
D2_NUM_CLASSES = svhn_d2_input.D2_NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 400#svhn_d2_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200#svhn_d2_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

TOWER_NAME = 'tower'
def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(images):
  """Build the svhn_d2 model.

  Args:
    images: Images returned from inputs().

  Returns:
    Logits.
  """
  # As we train this model on a single CPU, we instantiate all varianles using tf.Variable() instead of 
  # tf.get_variable(). 
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear_len') as scope:
    weights_len = _variable_with_weight_decay('weights_len', [192, L_NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases_len = _variable_on_cpu('biases_len', [L_NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear_len = tf.add(tf.matmul(local4, weights_len), biases_len, name=scope.name)
    _activation_summary(softmax_linear_len)

  with tf.variable_scope('softmax_linear_d1') as scope:
    weights_d1 = _variable_with_weight_decay('weights_d1', [192, D1_NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases_d1 = _variable_on_cpu('biases_d1', [D1_NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear_d1 = tf.add(tf.matmul(local4, weights_d1), biases_d1, name=scope.name)
    _activation_summary(softmax_linear_d1)

  with tf.variable_scope('softmax_linear_d2') as scope:
    weights_d2 = _variable_with_weight_decay('weights_d2', [192, D2_NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases_d2 = _variable_on_cpu('biases_d2', [D2_NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear_d2 = tf.add(tf.matmul(local4, weights_d2), biases_d2, name=scope.name)
    _activation_summary(softmax_linear_d2)

  return softmax_linear_len, softmax_linear_d1, softmax_linear_d2


def loss(len_logits, d1_logits, d2_logits, len_labels, d1_labels, d2_labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch
  cross_entropy_len = tf.nn.sparse_softmax_cross_entropy_with_logits(
      len_logits, len_labels, name='cross_entropy_per_example_len')
  cross_entropy_mean_len = tf.reduce_mean(cross_entropy_len, name='mean_cross_entropy_len')
  tf.add_to_collection('losses', cross_entropy_mean_len)
  # Calculate the average cross entropy loss across the batch.
  cross_entropy_d1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      d1_logits, d1_labels, name='cross_entropy_per_example_d1')

  cross_entropy_d2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      d2_logits, d2_labels, name='cross_entropy_per_example_d2')
  
  cross_entropy_2d = tf.pack([cross_entropy_d1, cross_entropy_d2])
  digits_len = tf.pack([tf.ones([FLAGS.batch_size], tf.float32), tf.cast(len_labels, tf.float32)])
  cross_entropy_2d = tf.multiply(cross_entropy_2d, digits_len)
  cross_entropy_2d = tf.reduce_sum(cross_entropy_2d, 0)
  cross_entropy_mean_2d = tf.reduce_mean(cross_entropy_2d, name='mean_cross_entropy_2_digits')
  tf.add_to_collection('losses', cross_entropy_mean_2d)
  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in this model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name + ' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train svhn_d2 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

