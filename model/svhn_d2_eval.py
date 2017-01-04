from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import svhn_d2
from svhn_d2_input import inputs

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/lwy/Downloads/svhn_data_2_32by32/test_log',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', '/home/lwy/Downloads/svhn_data_2_32by32/test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/lwy/Downloads/svhn_data_2_32by32/train_log',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, len_labels, l_top_k_op, d1_top_k_op, d2_top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    len_labels:
    l_top_k_op:
    d1_top_k_op: Top K op for the first digit.
    d2_top_k_op: Top K op for the second digit.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        batch_len_count = 0
        batch_d1_count = 0
        batch_d2_count = 0
        batch_count = 0
        two_d_count = 0
        lengths, len_predictions, d1_predictions, d2_predictions  = sess.run([len_labels, l_top_k_op, d1_top_k_op, d2_top_k_op])
        lengths = np.array(lengths).astype(float)
        len_predictions = np.array(len_predictions).astype(float)
        d1_predictions = np.array(d1_predictions).astype(float) 
        d2_predictions = np.array(d2_predictions).astype(float)
        for i in xrange(len(lengths)):
            if lengths[i] > 0:
                batch_count += len_predictions[i]*d1_predictions[i]*d2_predictions[i]
                batch_d2_count += d2_predictions[i]
                two_d_count += 1.0
            else:
                batch_count += len_predictions[i]*d1_predictions[i]
        print("correct count for this batch")
        print(batch_count)
        print("    house number prediction accuracy: "+str(batch_count/FLAGS.batch_size))
        print("    length prediction accuracy: "+str(np.sum(len_predictions)/FLAGS.batch_size))
        print("    the first digit prediction accuracy: "+str(np.sum(d1_predictions)/FLAGS.batch_size))
        print("    the second digit predition accuracy: "+str(batch_d2_count/two_d_count))
        true_count += batch_count
        print("accumulated right count")
        print(true_count)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for svhn_d2
    images, len_labels, d1_labels, d2_labels = inputs('/home/lwy/Downloads/svhn_data_2_32by32/test_labels','/home/lwy/Downloads/svhn_data_2_32by32/test', FLAGS.batch_size)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    len_digits, d1_logits, d2_logits = svhn_d2.inference(images)

    # Calculate predictions.
    l_top_k_op = tf.nn.in_top_k(len_digits, len_labels, 1)
    d1_top_k_op = tf.nn.in_top_k(d1_logits, d1_labels, 1)
    d2_top_k_op = tf.nn.in_top_k(d2_logits, d2_labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        svhn_d2.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, len_labels, l_top_k_op, d1_top_k_op, d2_top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
