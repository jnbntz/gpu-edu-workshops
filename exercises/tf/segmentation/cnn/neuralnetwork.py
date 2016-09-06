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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import tensorflow as tf

# function to print the tensor shape.  useful for debugging

def print_tensor_shape(tensor, string):

# input: tensor and string to describe it

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())


def read_and_decode(filename_queue):

# input: filename
# output: image, label pair

# setup a TF record reader
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

# list the features we want to extract, i.e., the image and the label
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string),
        })

  # Decode the training image
  # Convert from a scalar string tensor (whose single string has
  # length 256*256) to a float tensor
    image = tf.decode_raw(features['img_raw'], tf.int64)
    image.set_shape([65536])
    image_re = tf.reshape(image, (256,256))

# Scale input pixels by 1024
    image_re = tf.cast(image_re, tf.float32) * (1. / 1024)

# decode the label image, an image with all 0's except 1's where the left
# ventricle exists
    label = tf.decode_raw(features['label_raw'], tf.uint8)
    label.set_shape([65536])
    label_re = tf.reshape(label, [256,256])

    return image_re, label_re


def inputs(batch_size, num_epochs, filename):

# inputs: batch_size, num_epochs are scalars, filename
# output: image and label pairs for use in training or eval

    if not num_epochs: num_epochs = None

# define the input node
    with tf.name_scope('input'):

# setup a TF filename_queue
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

# return and image and label
        image, label = read_and_decode(filename_queue)
     
# shuffle the images, not strictly necessary as the data creating
# phase already did it, but there's no harm doing it again.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=15,
            min_after_dequeue = 10)

#        tf.image_summary( 'images', tf.reshape(images,[-1,256,256,1] ))
#        tf.image_summary( 'labels', tf.reshape(sparse_labels,[-1,256,256,1]))
        return images, sparse_labels


def inference(images):

#   input: tensor of images
#   output: tensor of computed logits

    print_tensor_shape( images, 'images shape inference' )

# resize the image tensors to add the number of channels, 1 in this case
# required to pass the images to various layers upcoming in the graph
    images_re = tf.reshape( images, [-1,256,256,1] ) 
    print_tensor_shape( images, 'images shape inference' )
    
# Convolution layer
    with tf.name_scope('Conv1'):

# weight variable 4d tensor, first two dims are patch (kernel) size       
# third dim is number of input channels and fourth dim is output channels
# will be convolved with images_re

        W_conv1 = tf.Variable(tf.truncated_normal([FIXME,FIXME,1,100],
                     stddev=0.1,
                     dtype=tf.float32),name='W_conv1')
        print_tensor_shape( W_conv1, 'W_conv1 shape')

# convolution operator.  first arg is the batch of input images with 
# shape [batch, in_height, in_width, in_channels]

# second arg is the filter (weights) with shape 
# [filter_height, filter_width, in_channels, out_channels]

# strides is a 4d tensor.  stride of the sliding window for each
# dimension of input

        conv1_op = tf.nn.conv2d( images_re, W_conv1, 
                     strides=[1,FIXME,FIXME,1], 
                     padding="SAME", name='conv1_op' )
        print_tensor_shape( conv1_op, 'conv1_op shape')

# rectified linear activation function

        relu1_op = tf.nn.relu( conv1_op, name='relu1_op' )
        print_tensor_shape( relu1_op, 'relu1_op shape')

# Pooling layer
    with tf.name_scope('Pool1'):

# max pooling layer
# ksize = size of the window for each input dimension
# strides = stride of the sliding window for each input dimension

        pool1_op = tf.nn.max_pool(relu1_op, ksize=[1,FIXME,FIXME,1],
                                  strides=[1,FIXME,FIXME,1], padding='SAME') 
        print_tensor_shape( pool1_op, 'pool1_op shape')

# Conv layer
    with tf.name_scope('Conv2'):
        W_conv2 = tf.Variable(tf.truncated_normal([FIXME,FIXME,100,200],
                     stddev=0.1,
                     dtype=tf.float32),name='W_conv2')
        print_tensor_shape( W_conv2, 'W_conv2 shape')

        conv2_op = tf.nn.conv2d( pool1_op, W_conv2, 
                     strides=[1,FIXME,FIXME,1],
                     padding="SAME", name='conv2_op' )
        print_tensor_shape( conv2_op, 'conv2_op shape')

        relu2_op = tf.nn.relu( conv2_op, name='relu2_op' )
        print_tensor_shape( relu2_op, 'relu2_op shape')

# Pooling layer
    with tf.name_scope('Pool2'):
        pool2_op = tf.nn.max_pool(relu2_op, ksize=[1,FIXME,FIXME,1],
                                  strides=[1,FIXME,FIXME,1], padding='SAME')
        print_tensor_shape( pool2_op, 'pool2_op shape')
    
# Conv layer
    with tf.name_scope('Conv3'):
        W_conv3 = tf.Variable(tf.truncated_normal([FIXME,FIXME,200,300],
                     stddev=0.1,
                     dtype=tf.float32),name='W_conv3') 
        print_tensor_shape( W_conv3, 'W_conv3 shape')

        conv3_op = tf.nn.conv2d( pool2_op, W_conv3, 
                     strides=[1,FIXME,FIXME,1],
                     padding='SAME', name='conv3_op' )
        print_tensor_shape( conv3_op, 'conv3_op shape')

        relu3_op = tf.nn.relu( conv3_op, name='relu3_op' )
        print_tensor_shape( relu3_op, 'relu3_op shape')
    
# Conv layer
    with tf.name_scope('Conv4'):
        W_conv4 = tf.Variable(tf.truncated_normal([FIXME,FIXME,300,300],
                    stddev=0.1,
                    dtype=tf.float32), name='W_conv4')
        print_tensor_shape( W_conv4, 'W_conv4 shape')

        conv4_op = tf.nn.conv2d( relu3_op, W_conv4, 
                     strides=[1,FIXME,FIXME,1],
                     padding='SAME', name='conv4_op' )
        print_tensor_shape( conv4_op, 'conv4_op shape')

        relu4_op = tf.nn.relu( conv4_op, name='relu4_op' )
        print_tensor_shape( relu4_op, 'relu4_op shape')

# optional dropout node.  when set to 1.0 nothing is dropped out
        drop_op = tf.nn.dropout( relu4_op, 1.0 )
        print_tensor_shape( drop_op, 'drop_op shape' )
    
# Conv layer to generate the 2 score classes
    with tf.name_scope('Score_classes'):
        W_score_classes = tf.Variable(tf.truncated_normal([1,1,300,2],
                            stddev=0.1,dtype=tf.float32),name='W_score_classes')
        print_tensor_shape( W_score_classes, 'W_score_classes_shape')

        score_classes_conv_op = tf.nn.conv2d( drop_op, W_score_classes, 
                       strides=[1,1,1,1], padding='SAME', 
                       name='score_classes_conv_op')
        print_tensor_shape( score_classes_conv_op,'score_conv_op shape')

# Upscore the results to 256x256x2 image
    with tf.name_scope('Upscore'):
        W_upscore = tf.Variable(tf.truncated_normal([31,31,2,2],
                              stddev=0.1,dtype=tf.float32),name='W_upscore')
        print_tensor_shape( W_upscore, 'W_upscore shape')
      
# conv2d_transpose is also referred to in the literature as 
# deconvolution
        upscore_conv_op = tf.nn.conv2d_transpose( score_classes_conv_op, 
                       W_upscore,
                       output_shape=[1,256,256,2],strides=[1,16,16,1],
                       padding='SAME',name='upscore_conv_op')
        print_tensor_shape(upscore_conv_op, 'upscore_conv_op shape')

    return upscore_conv_op

    
def loss(logits, labels):
    
    # input: logits: Logits tensor, float - [batch_size, 256, 256, NUM_CLASSES].
    # intput: labels: Labels tensor, int32 - [batch_size, 256, 256].
    # output: loss: Loss tensor of type float.

    labels = tf.to_int64(labels)
    print_tensor_shape( logits, 'logits shape before')
    print_tensor_shape( labels, 'labels shape before')

# reshape to match args required for the cross entropy function
    logits_re = tf.reshape( logits, [-1, 2] )
    labels_re = tf.reshape( labels, [-1] )
    print_tensor_shape( logits, 'logits shape after')
    print_tensor_shape( labels, 'labels shape after')

# call cross entropy with logits
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
         logits, labels, name='cross_entropy')

    loss = tf.reduce_mean(cross_entropy, name='1cnn_cross_entropy_mean')
    return loss


def training(loss, learning_rate, decay_steps, decay_rate):
    # input: loss: loss tensor from loss()
    # input: learning_rate: scalar for gradient descent
    # output: train_op the operation for training

#    Creates a summarizer to track the loss over time in TensorBoard.

#    Creates an optimizer and applies the gradients to all trainable variables.

#    The Op returned by this function is what must be passed to the
#    `sess.run()` call to cause the model to train.

  # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)

  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

  # create learning_decay
    lr = tf.train.exponential_decay( learning_rate,
                                     global_step,
                                     decay_steps,
                                     decay_rate, staircase=True )

    tf.scalar_summary('1learning_rate', lr )

  # Create the gradient descent optimizer with the given learning rate.
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(lr)

  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(logits, labels):
    # input: logits: Logits tensor, float - [batch_size, 256, 256, NUM_CLASSES].
    # input: labels: Labels tensor, int32 - [batch_size, 256, 256]
    # output: scaler int32 tensor with number of examples that were 
    #         predicted correctly

    with tf.name_scope('eval'):
        labels = tf.to_int64(labels)
        print_tensor_shape( logits, 'logits eval shape before')
        print_tensor_shape( labels, 'labels eval shape before')

# reshape to match args required for the cross entropy function
        logits_re = tf.reshape( logits, [-1, 2] )
        labels_re = tf.reshape( labels, [-1] )
        print_tensor_shape( logits, 'logits eval shape after')
        print_tensor_shape( labels, 'labels eval shape after')

  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
        correct = tf.nn.in_top_k(logits_re, labels_re, 1)
        print_tensor_shape( correct, 'correct shape')

  # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))
