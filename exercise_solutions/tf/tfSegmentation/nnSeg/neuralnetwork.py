# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import tensorflow as tf

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
#  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(x.op.name + '/activations', x)
  tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

def inference(images):
    """Build the MNIST model up to where it may be used for inference.

    Args:
        images: the images from the input

    Returns:
        softmax_linear: Output tensor with the computed logits.
    """
    print('images shape', images.get_shape())
# resize the image tensors to add the number of channels (1) 
    images_re = tf.reshape( images, [-1,256,256,1] ) 
    print('images after reshape',images_re.get_shape())
    
# Convolution 1
    with tf.variable_scope('Conv1'):
# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
        W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,100],stddev=0.1,
                     dtype=tf.float32),name='W_conv1')
        print("W_conv1 shape",W_conv1.get_shape())

        conv1_op = tf.nn.conv2d( images_re, W_conv1, strides=[1,2,2,1], 
                     padding="SAME", name='conv1_op' )
        print("conv1_op shape",conv1_op.get_shape())
        _activation_summary(conv1_op)

        relu1_op = tf.nn.relu( conv1_op, name='relu1_op' )
        print("relu1_op shape",relu1_op.get_shape())

    with tf.name_scope('Pool1'):
        pool1_op = tf.nn.max_pool(relu1_op, ksize=[1,2,2,1],
                                  strides=[1,2,2,1], padding='SAME') 
        print("pool1_op shape",pool1_op.get_shape())

    with tf.name_scope('Conv2'):
        W_conv2 = tf.Variable(tf.truncated_normal([5,5,100,200],stddev=0.1,
                     dtype=tf.float32),name='W_conv2')
        print("W_conv2 shape",W_conv2.get_shape())

        conv2_op = tf.nn.conv2d( pool1_op, W_conv2, strides=[1,2,2,1],
                     padding="SAME", name='conv2_op' )
        print("conv2_op shape",conv2_op.get_shape())

        relu2_op = tf.nn.relu( conv2_op, name='relu2_op' )
        print("relu2_op shape",relu2_op.get_shape())

    with tf.name_scope('Pool2'):
        pool2_op = tf.nn.max_pool(relu2_op, ksize=[1,2,2,1],
                                  strides=[1,2,2,1], padding='SAME')
        print("pool2_op shape",pool2_op.get_shape())
    
    with tf.name_scope('Conv3'):
        W_conv3 = tf.Variable(tf.truncated_normal([3,3,200,300],stddev=0.1,
                     dtype=tf.float32),name='W_conv3') 
        print("W_conv3 shape",W_conv3.get_shape())

        conv3_op = tf.nn.conv2d( pool2_op, W_conv3, strides=[1,1,1,1],
                     padding='SAME', name='conv3_op' )
        print("conv3_op shape",conv3_op.get_shape())

        relu3_op = tf.nn.relu( conv3_op, name='relu3_op' )
        print("relu3_op shape",relu3_op.get_shape())
    
    with tf.name_scope('Conv4'):
        W_conv4 = tf.Variable(tf.truncated_normal([3,3,300,300],stddev=0.1,
                    dtype=tf.float32), name='W_conv4')
        print("W_conv4 shape",W_conv4.get_shape())

        conv4_op = tf.nn.conv2d( relu3_op, W_conv4, strides=[1,1,1,1],
                     padding='SAME', name='conv4_op' )
        print("conv4_op shape",conv4_op.get_shape())

        drop_op = tf.nn.dropout( conv4_op, 1.0 )
        print("drop_op shape",drop_op.get_shape())
    
    with tf.name_scope('Score_classes'):
        W_score_classes = tf.Variable(tf.truncated_normal([1,1,300,2],
                            stddev=0.1,dtype=tf.float32),name='W_score_classes')
        print("W_score_classes shape",W_score_classes.get_shape())

        score_classes_conv_op = tf.nn.conv2d( drop_op, W_score_classes, 
                       strides=[1,1,1,1], padding='SAME', 
                       name='score_classes_conv_op')
        print("score_conv_op shape",score_classes_conv_op.get_shape())

    with tf.name_scope('Upscore'):
        W_upscore = tf.Variable(tf.truncated_normal([31,31,2,2],
                              stddev=0.1,dtype=tf.float32),name='W_upscore')
        print("W_upscore shape",W_upscore.get_shape())
      
        upscore_conv_op = tf.nn.conv2d_transpose( score_classes_conv_op, 
                       W_upscore,
                       output_shape=[1,256,256,2],strides=[1,16,16,1],
                       padding='SAME',name='upscore_conv_op')
        print("upscore_conv_op shape",upscore_conv_op.get_shape())

    return upscore_conv_op

    
def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
        logits: Logits tensor, float - [batch_size, 256, 256, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size, 256, 256].

    Returns:
        loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    print("logits shape before", logits.get_shape())
    print("labels shape before", labels.get_shape())

# reshape to match args required for the cross entropy function
    logits_re = tf.reshape( logits, [-1, 2] )
    labels_re = tf.reshape( labels, [-1] )
    print("logits shape after", logits_re.get_shape())
    print("labels shape after", labels_re.get_shape())

# call cross entropy with logits
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
         logits, labels, name='cross_entropy')

    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return loss


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.

    Returns:
        train_op: The Op for training.
    """
  # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
           range [0, NUM_CLASSES).

    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """ 
    labels = tf.to_int64(labels)
    print("logits eval shape before", logits.get_shape())
    print("labels eval shape before", labels.get_shape())

# reshape to match args required for the cross entropy function
    logits_re = tf.reshape( logits, [-1, 2] )
    labels_re = tf.reshape( labels, [-1] )
    print("logits eval shape after", logits_re.get_shape())
    print("labels eval shape after", labels_re.get_shape())
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
    correct = tf.nn.in_top_k(logits_re, labels_re, 1)
  # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
