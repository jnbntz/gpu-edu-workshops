
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
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('train_dir', '/tmp/data',
                    'Directory with the training data.')
def run_training():
 
    with tf.Graph().as_default():

        trainfile = os.path.join(FLAGS.train_dir, TRAIN_FILE)

        images, labels = nn.inputs(train=True, batch_size=FLAGS.batch_size,
                                num_epochs=FLAGS.num_epochs,
                                filename=trainfile)

        results = nn.inference(images)

        loss = nn.loss(results, labels)

        train_op = nn.training(loss, FLAGS.learning_rate)

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
    for serialized_example in tf.python_io.tf_record_iterator('/tmp/data/train_images.tfrecords'):
        idx += 1
    print("idx is %d" % idx)    
    idx = 0
    for serialized_example in tf.python_io.tf_record_iterator('/tmp/data/val_images.tfrecords'):
        idx += 1
    print("idx is %d" % idx)    
    run_training()

if __name__ == '__main__':
    tf.app.run()
