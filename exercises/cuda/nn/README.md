Instructions
------------

To run the code which trains and then classifies a handwritten digit do the
following steps.

1.) Grab the MNIST files from the Yann Lecun's website.

> sh setupData.sh

2.) Build the code.  Ensure that nvcc is in your path.

> make

3.) Run the code.  In this step the network will be trained on the 60,000 
images from MNIST then compared against 10,000 test images.

./x.nn

Learning rate lambda is               3.000e-01
Batchsize is                          50
Number of iterations is               1
Hidden Layer Size is                  25
Number of training examples           60000
Number of features/pixels per example 784
Number of test examples               10000
|
Total time for training is            1.277e+00 sec
Total correct on training set is      48960
Prediction rate of training set is    81.600
Total correct on test set is          8214
Prediction rate of test set is        82.140
 
