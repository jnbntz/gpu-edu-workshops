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

# Choose/change the hidden layer dimension
    hidden = 512

    with tf.name_scope('Hidden1'):

# Create the weights matrix for the first layer
        W_fc = tf.Variable(tf.truncated_normal( [256*256, hidden],
                     stddev=0.1, dtype=tf.float32), name='W_fc')
        print_tensor_shape( W_fc, 'W_fc shape')

# flatten the tensor to align with arg requirements for matmul
        flatten1_op = tf.reshape( images_re, [-1, 256*256])
        print_tensor_shape( flatten1_op, 'flatten1_op shape')

# multiple the images with the weights
        h_fc1 = tf.matmul( flatten1_op, W_fc )
        print_tensor_shape( h_fc1, 'h_fc1 shape')
    
    with tf.name_scope('Final'):

# create weights matrix for second layer
        W_fc2 = tf.Variable(tf.truncated_normal( [hidden, 256*256*2],
                    stddev=0.1, dtype=tf.float32), name='W_fc2' )
        print_tensor_shape( W_fc2, 'W_fc2 shape')

# matrix multiply them together
        h_fc2 = tf.matmul( h_fc1, W_fc2 )
        print_tensor_shape( h_fc2, 'h_fc2 shape')

# reshape the output back to a 2d image
        h_fc2_re = tf.reshape( h_fc2, [-1, 256, 256, 2] )
        print_tensor_shape( h_fc2_re, 'h_fc2_re shape')
        
    return h_fc2_re 

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
         labels=labels, logits=logits, name='cross_entropy')
    print_tensor_shape( cross_entropy, 'cross_entropy shape')

    loss = tf.reduce_mean(cross_entropy, name='0simple_cross_entropy_mean')
    return loss


def training(loss, learning_rate):
    # input: loss: loss tensor from loss()
    # input: learning_rate: scalar for gradient descent
    # output: train_op the operation for training

#    Creates a summarizer to track the loss over time in TensorBoard.

#    Creates an optimizer and applies the gradients to all trainable variables.

#    The Op returned by this function is what must be passed to the
#    `sess.run()` call to cause the model to train.

  # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(loss.op.name, loss)

  # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

  # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

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
        print_tensor_shape( logits_re, 'logits_re eval shape after')
        print_tensor_shape( labels_re, 'labels_re eval shape after')

  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
        correct = tf.nn.in_top_k(logits_re, labels_re, 1)
        print_tensor_shape( correct, 'correct shape')

  # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))
