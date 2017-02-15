import numpy as np
import os
import sys
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

def main(argv):

    path='http://yann.lecun.com/exdb/mnist/'
    (train_lbl, train_img) = read_data(
        path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
        path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

    batch_size = 100

    train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, 
                 shuffle=True)

    val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

    model_prefix = './model/mnist'

    sym, arg_params, aux_params = mx.model.load_checkpoint(
                                  model_prefix,
                                  30)
    
    model = mx.mod.Module(symbol=sym, context=mx.gpu(0))

    model.bind(for_training=False, data_shapes=val_iter.provide_data,
               label_shapes=val_iter.provide_label)

    model.set_params(arg_params, aux_params)

    valid_acc = model.score(val_iter, mx.metric.Accuracy())
    print 'Validation accuracy: %s' % valid_acc
    assert valid_acc > 0.95, "Low validation accuracy."

    train_acc = model.score(train_iter, mx.metric.Accuracy())
    print 'Training accuracy: %s' % train_acc
    assert train_acc > 0.95, "Low validation accuracy."

if __name__ == "__main__":
    main(sys.argv)
