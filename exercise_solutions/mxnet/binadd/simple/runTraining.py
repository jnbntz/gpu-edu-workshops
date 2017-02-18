import sys
import copy
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp( -x ) )

def sigmoid_derivative(x):
    return sigmoid(x) * ( 1 - sigmoid(x) )

def sigmoid_output_to_derivative(x):
    return x * ( 1 - x )

def print_binary(num, dim):
    length = len( np.binary_repr( num ) )
#    length = len( str(num) )
    assert length <= dim
    leading_zeros = dim - length
    for index in range(leading_zeros):
         print '0',
    print str(np.binary_repr( num ))

def to_bin_array(num, dim):
    arr = [int(i) for i in np.binary_repr(num)]
    length = len(arr)
    for index in range(dim - length):
        arr.insert(0, int(0))
    arr = np.asarray(arr)
    return arr

def gen_input(largest):
    a_int = np.random.randint(largest/2)
    b_int = np.random.randint(largest/2)
    return a_int, b_int

def predict( X, w_0, w_h, w_1, layer_1_values ):
# sigmoid activation of the input times first weight matrix
# vector add elementwise with the prior iteration layer 1 values and 
# the hidden weight matrix
    layer_1 = sigmoid( np.dot(X, w_0) + np.dot( layer_1_values[-1], w_h ))

# take the result of the inner nodes, multiply against the next weight matrix
# and take the activation with sigmoid
    layer_2 = sigmoid( np.dot( layer_1, w_1 ) )

# output the value.  It's a 1,1 matrix
    return np.round( layer_2[0][0] )


np.random.seed(0)

binary_dim = 8

largest_number = pow(2, binary_dim)

print to_bin_array(53, 8)

print "The binary representation of 53 is " + str(to_bin_array(53,8))

a_int, b_int = gen_input(largest_number)

a = to_bin_array(a_int, binary_dim)
print "a = " + str(a)

b = to_bin_array(b_int, binary_dim)
print "b = " + str(b)

c_int = a_int + b_int
c = to_bin_array(c_int, binary_dim)
print "c = " + str(c)

d = np.zeros_like(c)

input_dim = 2
hidden_dim = 16
output_dim = 1

w_0 = 2*np.random.random( (input_dim, hidden_dim) ) - 1
w_1 = 2*np.random.random( (hidden_dim, output_dim) ) - 1
w_h = 2*np.random.random( (hidden_dim, hidden_dim) ) - 1

w_0_update = np.zeros_like( w_0 )
w_1_update = np.zeros_like( w_1 )
w_h_update = np.zeros_like( w_h )

a_int, b_int = gen_input(largest_number)

a = to_bin_array(a_int, binary_dim)
print "a = " + str(a)

b = to_bin_array(b_int, binary_dim)
print "b = " + str(b)

c_int = a_int + b_int
c = to_bin_array(c_int, binary_dim)
print "c (Truth)      = " + str(c)

d = np.zeros_like(c)

layer_1_values = list()
layer_1_values.append(np.zeros(hidden_dim))


for position in range(binary_dim-1,-1,-1):

# grab a pair of digits from the two input numbers
# create as column vector for for proper dot product notations
    X = np.array([[a[position],b[position]]])
    
# sigmoid activation of the input times first weight matrix
# vector add elementwise with the prior iteration layer 1 values and 
# the hidden weight matrix
#    layer_1 = sigmoid( np.dot(X, w_0) + np.dot( layer_1_values[-1], w_h ))

# take the result of the inner nodes, multiply against the next weight matrix
# and take the activation with sigmoid
#    layer_2 = sigmoid( np.dot( layer_1, w_1 ) )

# output the value.  It's a 1,1 matrix
#    d[position] = np.round( layer_2[0][0] )
    d[position]  = predict( X, w_0, w_h, w_1, layer_1_values )

print "d (Prediction) = " + str(d)

alpha = 0.1

for num_epochs in range(100000):
    a_int, b_int = gen_input(largest_number)

    a = to_bin_array(a_int, binary_dim)
    b = to_bin_array(b_int, binary_dim)

    c_int = a_int + b_int
    c = to_bin_array(c_int, binary_dim)

    d = np.zeros_like(c)
    overall_error = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

# count backwards through the binary digits
    for position in range(binary_dim-1, -1, -1):
        X = np.array([[a[position], b[position]]])
        y = np.array([[c[position]]]).T

# multiply input by first weight matrix
#        X_times_w_0 = np.dot(X,w_0)

# multiply previous hidden layer
# -1 means last element of the array
#        temp = np.dot( layer_1_values[-1], w_h )

# add current input * weight with previous hidden layer
#        temp1 = X_times_w_0 + temp
    
# sigmoid activation
        layer_1 = sigmoid( np.dot(X, w_0) + np.dot(layer_1_values[-1], w_h) )

# take the result of the inner nodes, dot product with next weight matrix
# and take the activation with sigmoid
        layer_2 = sigmoid( np.dot( layer_1, w_1 ) )

#        d[position], layer_1, layer_2 = predict( X, w_0, w_h, w_1, layer_1_values )
        d[position] = np.round(layer_2[0][0])

        layer_2_error = y - layer_2
        layer_2_deltas.append(
            (layer_2_error)*sigmoid_output_to_derivative(layer_2))
        
        overall_error += np.abs(layer_2_error[0])


        layer_1_values.append(copy.deepcopy(layer_1))

#    print layer_1_values[7].shape
    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]

        layer_2_delta = layer_2_deltas[-position-1]

        layer_1_delta = (future_layer_1_delta.dot(w_h.T) + 
                        layer_2_delta.dot(w_1.T))  \
                        * sigmoid_output_to_derivative(layer_1)

        w_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        w_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        w_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    w_0 += w_0_update * alpha
    w_1 += w_1_update * alpha
    w_h += w_h_update * alpha

    w_0_update *= 0
    w_1_update *= 0
    w_h_update *= 0

    if( num_epochs % 500 == 0 ):
        alpha *= 1.0
        print "Epoch:" + str(num_epochs)
        print "Error:" + str(overall_error)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        if a_int + b_int == out:
            correct = "TRUE"
        else:
            correct = "FALSE"
        print str(a_int) + " + " + str(b_int) + " = " + str(out) + " " + correct
        print "------------"
