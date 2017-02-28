#
# Copyright 2017 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from datetime import datetime
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork as nn

# set the training and validation file names

TRAIN_FILE = 'train_images.tfrecords'
VALIDATION_FILE = 'val_images.tfrecords'

# flags is a TensorFlow way to manage command line flags.

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('data_dir', '/tmp/sunny_data',
                    'Directory with the image data.')
flags.DEFINE_string('eval_dir', '/tmp/sunny_eval',
                           """Directory where to write event logs.""")
flags.DEFINE_string('eval_data', 'eval',
                           """Either 'train' or 'eval', depending whether you want to evaluate the training set or validation set""")
flags.DEFINE_string('checkpoint_dir', '/tmp/sunny_train',
                           """Directory where to read model checkpoints.""")
flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

def run_eval():

    # Run evaluation on the input data set
    with tf.Graph().as_default() as g:

    # Get images and labels for the MRI data
        eval_data = FLAGS.eval_data == 'eval'

# choose whether to evaluate the training set or the evaluation set
        evalfile = os.path.join(FLAGS.data_dir, 
                    VALIDATION_FILE if eval_data else TRAIN_FILE)

# read the proper data set
        images, labels = nn.inputs(batch_size=FLAGS.batch_size,
                           num_epochs=1, filename=evalfile)

    # Build a Graph that computes the logits predictions from the
    # inference model.  We'll use a prior graph built by the training
        logits = nn.inference(images)

    # Calculate predictions.
        int_area, label_area, example_area = nn.evaluation(logits, labels)

    # setup the initialization of variables
        local_init = tf.local_variables_initializer()

    # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

# create the saver and session
        saver = tf.train.Saver()
        sess = tf.Session()

# init the local variables
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
 
# start up the threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:

# true_count accumulates the correct predictions
                int_sum = 0
                label_sum = 0
                example_sum = 0 
#                true_count = 0
                step = 0
                while not coord.should_stop():

# run a single iteration of evaluation
#                    predictions = sess.run([top_k_op])
                    ii, ll, ee = sess.run([int_area, label_area, example_area])
                    int_sum += ii
                    label_sum += ll
                    example_sum += ee
# aggregate correct predictions 
#                    true_count += np.sum(predictions)
                    step += 1

# uncomment below line for debugging
#                    print("step ii, ll, ee, iI, lL, eE", 
#                             step, ii, ll, ee, int_sum,
#                              label_sum, example_sum)
        
            except tf.errors.OutOfRangeError:
# print and output the relevant prediction accuracy
#                precision = true_count / ( step * 256.0 * 256 )
                precision = (2.0 * int_sum) / ( label_sum + example_sum )
                print('OUTPUT: %s: Dice metric = %.3f' % (datetime.now(), precision))
                print('OUTPUT: %d images evaluated from file %s' % (step, evalfile))

# create summary to show in TensorBoard
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='2Dice metric', simple_value=precision)
                summary_writer.add_summary(summary, global_step)

            finally:
                coord.request_stop()
        
# shutdown gracefully
            coord.join(threads)
             
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)
            sess.close()

def main(_):
    run_eval()

if __name__ == '__main__':
    tf.app.run()
