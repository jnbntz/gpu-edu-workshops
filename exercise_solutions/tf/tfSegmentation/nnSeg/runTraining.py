
import time
import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork as nn

TRAIN_FILE = 'train_images.tfrecords'
VALIDATION_FILE = 'val_images.tfrecords'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 20, 'Number of epochs to run trainer.')
#flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
#flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('train_dir', '/tmp/data',
                    'Directory with the training data.')


def read_and_decode(filename_queue):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
      # Defaults are not specified since both keys are required.
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['img_raw'], tf.int64)
    image.set_shape([65536])
    image_re = tf.reshape(image, (256,256))
    image_re = tf.cast(image_re, tf.float32) * (1. / 1024)
#
  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
#    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label.set_shape([65536])
    label_re = tf.reshape(label, (256,256))
#    print(image_re.get_shape())

  # Convert label from a scalar uint8 tensor to an int32 scalar.
#    label = tf.cast(features['label'], tf.int32)
    return image_re, label_re

def inputs(train, batch_size, num_epochs):

    if not num_epochs: num_epochs = None
    filename = os.path.join(FLAGS.train_dir, 
                            TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue)
     
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=200,
            min_after_dequeue = 10)
#        images, sparse_labels = tf.train.batch(
#            [image, label], batch_size=batch_size, num_threads=2,
#            capacity=300)

#        tf.image_summary( 'images', tf.reshape(images,[-1,256,256,1] ))
        tf.image_summary( 'labels', tf.reshape(sparse_labels,[-1,256,256,1]))
        return images, sparse_labels

def do_eval(sess, eval_correct, images, labels ):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
          input_data.read_data_sets().
  """
 # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
#  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = 256*256 #steps_per_epoch * FLAGS.batch_size
#  for step in xrange(steps_per_epoch):
#    feed_dict = fill_feed_dict(data_set,
#                               images_placeholder,
#                               labels_placeholder)
    true_count += sess.run(eval_correct)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
 
    with tf.Graph().as_default():

        images, labels = inputs(train=True, batch_size=FLAGS.batch_size,
                                num_epochs=FLAGS.num_epochs)

        results = nn.inference(images)

        loss = nn.loss(results, labels)

        train_op = nn.training(loss, FLAGS.learning_rate)

        eval_correct = nn.evaluation( results, labels )

        summary_op = tf.merge_all_summaries()

        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())

        saver = tf.train.Saver()
    
        sess = tf.Session()

        summary_writer = tf.train.SummaryWriter('/tmp/data', sess.graph)

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                
                duration = time.time() - start_time

                if step % 100 == 0:
                    print('Step %d: loss = %.3f (%.3f sec)' % (step, 
                                                               loss_value,
                                                               duration))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                if step % 1000 == 0:
                    correct = sess.run(eval_correct)
                    print("correct is ",correct)
#                    do_eval(sess, eval_correct, images, labels)
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,
                                                              step))
        finally:
            coord.request_stop()
    
        coord.join(threads)
        sess.close()


def main(_):
    idx = 0
#    for serialized_example in tf.python_io.tf_record_iterator('/tmp/data/train_images.tfrecords'):
#        idx += 1
#    print("idx is %d" % idx)    
#    image = 0
#    for serialized_example in tf.python_io.tf_record_iterator('/tmp/data/val_images.tfrecords'):
#        idx += 1
#    print("idx is %d" % idx)    
    run_training()

if __name__ == '__main__':
    tf.app.run()
