
from datetime import datetime
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
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('train_dir', '/tmp/data',
                    'Directory with the training data.')
flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

def predict(logits, labels):

# calculate the predictions
    labels = tf.to_int64(labels)

# reshape to match requirements for in_top_k
    logits_re = tf.reshape( logits, [-1, 2] )
    labels_re = tf.reshape( labels, [-1] )
    result = tf.nn.in_top_k(logits_re, labels_re, 1)
    return result

def evaluate():

    # Run evaluation on the input data set
    with tf.Graph().as_default() as g:

    # Get images and labels for the MRI data
        eval_data = FLAGS.eval_data == 'test'
        evalfile = os.path.join(FLAGS.train_dir, VALIDATION_FILE)
        images, labels = nn.inputs(train=False, batch_size=FLAGS.batch_size,
                           num_epochs=1, filename=evalfile)

    # Build a Graph that computes the logits predictions from the
    # inference model.  We'll use a prior graph built by the training
        logits = nn.inference(images)

    # Calculate predictions.
        top_k_op = predict(logits, labels)

        local_init = tf.initialize_local_variables()

    # Build the summary operation based on the TF collection of Summaries.

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        saver = tf.train.Saver()
        sess = tf.Session()

        sess.run(local_init)

        while True:
    # read in the most recent checkpointed graph and weights    
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)     
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found in %s' % FLAGS.checkpoint_dir)
                return
 
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                true_count = 0
                step = 0
                print("step truecount", step, true_count)
                print("coord.should_stop", coord.should_stop())
                while not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1
                    print("step truecount", step, true_count)

        
            except tf.errors.OutOfRangeError:
    # print and output the relevant prediction accuracy
                precision = true_count / ( step * 256.0 * 256 )
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
                print('%d images evaluated from file %s' % (step, evalfile))
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)

            finally:
                coord.request_stop()
        
            coord.join(threads)
             
#            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
            sess.close()

def main(_):
    evaluate()

if __name__ == '__main__':
    tf.app.run()
