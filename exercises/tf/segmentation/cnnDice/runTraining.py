#
# Copyright 2016 NVIDIA Corporation
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

import time
import os.path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork as nn

tf.logging.set_verbosity(tf.logging.DEBUG)

TRAIN_FILE = 'train_images.tfrecords'
VALIDATION_FILE = 'val_images.tfrecords'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('decay_rate', 1.0, 'Learning rate decay.')
flags.DEFINE_integer('decay_steps', 1000, 'Steps at each learning rate.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_string('data_dir', '/tmp/sunny_data',
                    'Directory with the training data.')
flags.DEFINE_string('checkpoint_dir', '/tmp/sunny_train',
                           """Directory where to write model checkpoints.""")


def run_training():
 
# construct the graph
    with tf.Graph().as_default():

# specify the training data file location
        trainfile = os.path.join(FLAGS.data_dir, TRAIN_FILE)

# read the images and labels
        images, labels = nn.inputs(batch_size=FLAGS.batch_size,
                                num_epochs=FLAGS.num_epochs,
                                filename=trainfile)

# run inference on the images
        results = nn.inference(images)

# calculate the loss from the results of inference and the labels
        loss = nn.loss(results, labels)

# setup the training operations
        train_op = nn.training(loss, FLAGS.learning_rate, FLAGS.decay_steps,
                       FLAGS.decay_rate)

# setup the summary ops to use TensorBoard
        summary_op = tf.merge_all_summaries()

# init to setup the initial values of the weights
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())

# setup a saver for saving checkpoints
        saver = tf.train.Saver()
    
# create the session
        sess = tf.Session()

# specify where to write the log files for import to TensorBoard
        summary_writer = tf.train.SummaryWriter(FLAGS.checkpoint_dir,  
                            sess.graph)

# initialize the graph
        sess.run(init_op)

# setup the coordinato and threadsr.  Used for multiple threads to read data.  
# Not strictly required since we don't have a lot of data but typically 
# using multiple threads to read data improves performance
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# loop will continue until we run out of input training cases
        try:
            step = 0
            while not coord.should_stop():

# start time and run one training iteration
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time

# print some output periodically
                if step % 100 == 0:
                    print('OUTPUT: Step %d: loss = %.3f (%.3f sec)' % (step, 
                                                               loss_value,
                                                               duration))
# output some data to the log files for tensorboard
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

# less frequently output checkpoint files.  Used for evaluating the model
                if step % 1000 == 0:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 
                                                     'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                step += 1

# quit after we run out of input files to read
        except tf.errors.OutOfRangeError:
            print('OUTPUT: Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,
                                                              step))
            checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 
                                              'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

        finally:
            coord.request_stop()
    
# shut down the threads gracefully
        coord.join(threads)
        sess.close()


def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
