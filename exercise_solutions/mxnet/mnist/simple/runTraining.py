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

import numpy as np
import os
import sys
import argparse
import urllib
import gzip
import struct
import mxnet as mx
import logging

logging.getLogger().setLevel(logging.DEBUG)

def download_data(url, force_download=True): 
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), 
                rows, cols)
    return (label, image)

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255



#import matplotlib.pyplot as plt
#for i in range(10):
#    plt.subplot(1,10,i+1)
#    plt.imshow(train_img[i], cmap='Greys_r')
#    plt.axis('off')
#plt.show()
#print('label: %s' % (train_lbl[0:10],))

def train(batch_size, hidden_layer, num_epochs, activation):

    path='http://yann.lecun.com/exdb/mnist/'
    (train_lbl, train_img) = read_data(
        path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
        path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

#    batch_size = 100

    train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, 
                 shuffle=True)

    val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

    data = mx.sym.Variable('data', init=mx.initializer.Zero())

    data = mx.sym.Flatten(data=data)

    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=hidden_layer)

    act1 = mx.sym.Activation(data=fc1, name='sigmoid', act_type=activation)

    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=10)

    mlp = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

    model = mx.mod.Module( mlp, context=mx.gpu(0) )

    if not os.path.exists('model'):
        os.makedirs('./model')

    model_prefix = './model/mnist'
    checkpoint = mx.callback.do_checkpoint(model_prefix)

    model.fit(
        train_data=train_iter,
        eval_data=val_iter,
        num_epoch=num_epochs,
        optimizer_params={'learning_rate':0.1},
        batch_end_callback = mx.callback.Speedometer(batch_size, 200),
        epoch_end_callback = checkpoint
    )

    valid_acc = model.score(val_iter, mx.metric.Accuracy())
    print 'Validation accuracy: %s' % valid_acc
    assert valid_acc > 0.95, "Low validation accuracy."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an MLP on MNIST data')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-layer', type=int, default=25)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--activation', type=str, default='sigmoid',
        help='activation function')
    args = parser.parse_args()
    train(**vars(args))
#    main(sys.argv)
