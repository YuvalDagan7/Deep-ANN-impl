from datetime import datetime
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

OnTrain: bool
UseDropout: bool
UseBatchnorm: bool

def split_train_validation(x_train, y_train):
    train_size = x_train.shape[1]
    indices = np.random.permutation(train_size)

    x_train, y_train = x_train[:, indices], y_train[:, indices]

    # training_idx, validation_idx = indices[:int(train_size * 0.8)], indices[int(train_size * 0.8):]

    x_train, x_validation = x_train[:, :int(train_size * 0.8)], x_train[:, int(train_size * 0.8):]
    y_train, y_validation = y_train[:, :int(train_size * 0.8)], y_train[:, int(train_size * 0.8):]

    return x_train, x_validation, y_train, y_validation

def load_mnist():
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = x_train.reshape(x_train.shape[0], (x_train.shape[1]*x_train.shape[2])).transpose()
    x_test = x_test.reshape(x_test.shape[0], (x_test.shape[1] * x_test.shape[2])).transpose()

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes).transpose()
    y_test = keras.utils.to_categorical(y_test, num_classes).transpose()

    return x_train, x_test, y_train, y_test

# Forward propagation

def apply_dropout(keep_prob, A):
    d = (np.random.rand(*A.shape) < keep_prob) / keep_prob
    A = np.multiply(A, d)
    return A, d

def initialize_parameters(layer_dims: list):
    params = {}
    for i in range(len(layer_dims) - 1):
        # params['W' + str(i + 1)] = np.random.normal(1, 1, size=(layer_dims[i+1], layer_dims[i]))
        params['W' + str(i + 1)] = np.random.randn(layer_dims[i+1], layer_dims[i])/ np.sqrt(layer_dims[i])
        params['b' + str(i + 1)] = np.zeros((layer_dims[i + 1],1))

    return params

def linear_forward(A, W, b):
    z = np.dot(W, A) + b
    return z, {'A': A, 'W': W, 'b': b}

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return (expZ / expZ.sum(axis=0, keepdims=True)), {'Z': Z}

def relu(Z):
    res = Z * (Z > 0)
    return res, {'Z': Z}

def linear_activation_forward(A_prev, W, B, activation):
    cache = {}

    z, params = linear_forward(A_prev, W, B)

    cache.update(params)

    if activation == 'relu':
        A, params = relu(z)
    else:
        A, params = softmax(z)

    cache.update(params)
    return A, cache

def L_model_forward(X, parameters, use_batchnorm):
    global UseBatchnorm
    UseBatchnorm = use_batchnorm
    num_of_layers = parameters['layers_dim']
    caches = {}
    A = X
    for i in range(1, num_of_layers + 1):
        if i == num_of_layers:
            A, params = linear_activation_forward(A, parameters['W' + str(i)], parameters['b' + str(i)], 'softmax')
        else:
            A, params = linear_activation_forward(A, parameters['W' + str(i)], parameters['b' + str(i)], 'relu')
            if OnTrain and UseDropout:
                A, mask = apply_dropout(0.9, A)
                caches['mask' + str(i)] = mask
            else:
                caches['mask' + str(i)] = None
            if use_batchnorm:
                A = apply_batchnorm(A)
        caches['Z' + str(i)] = params['Z']
        caches['A' + str(i)] = A

    AL = A
    return AL, caches

# AL: (num_of_classes, num_of_examples)
# Y: (num_of_classes, num_of_examples)
def compute_cost(AL, Y):
    res = np.multiply(AL, Y)
    res = res[np.nonzero(res)]
    loss = -np.log(res + 1e-8).sum()

    m = Y.shape[1]
    loss = loss / m
    return loss

def apply_batchnorm(Z):
    mean = np.mean(Z, axis=1, keepdims=True)
    var = np.var(Z, axis=1, keepdims=True)
    std = np.sqrt(var + 1e-12)

    mean = Z - mean
    return mean / std

# Back propagation

def linear_backward(dZ, cache):
    params = {}
    m = cache['size_of_batch']
    params['dW'] = np.dot(dZ, cache['A'].T) / m
    params['db'] = np.sum(dZ, axis=1, keepdims=True) / m
    params['dA_prev'] = np.dot(cache['W'].T, dZ)
    return params

# dropout backprop
def dropout_backward(dA, activation_cache):
    if activation_cache['mask'] is None:
        return dA
    else:
        return dA * activation_cache['mask']

def linear_activation_backward(dA, cache, activation): #a = activation(z) -> dZ, dW,dA,db
    if activation == 'relu':
        dZ = dropout_backward(dA, cache)
        dZ = relu_backwards(dZ, cache)
    else:
        dZ = softmax_backward(dA, cache)

    return linear_backward(dZ, cache)

def relu_backwards(dA, activation_cache): #dL/dz = dL/dA * dA/dZ
    Z = activation_cache['Z']
    Z[Z > 0] = 1
    Z[Z < 0] = 0
    return dA * Z

def softmax_backward(dA, activation_cache):
    return activation_cache['AL'] - activation_cache['Y']

# AL: (num_of_classes, num_of_examples)
def L_model_backward(AL, Y, caches):
    grads = {}
    num_of_layers = 4

    params = linear_activation_backward(1, {'Y': Y, 'AL':AL, 'W': caches['W' + str(num_of_layers)],
                                            'A': caches['A' + str(num_of_layers-1)], 'size_of_batch':caches['size_of_batch']}, 'softmax')

    dA = params['dA_prev']
    grads['dW' + str(num_of_layers)] = params['dW']
    grads['db' + str(num_of_layers)] = params['db']

    for j in range(num_of_layers - 1, 0, -1):
        params = linear_activation_backward(dA, {'W': caches['W' + str(j)], 'A': caches['A' + str(j-1)], 'mask': caches['mask'+str(j)],
                                                'Z': caches['Z' + str(j)],'size_of_batch':caches['size_of_batch']}, 'relu')
        # dA = grads['dA' + str(num_of_layers)] = params['dA_prev']
        dA = params['dA_prev']
        grads['dW' + str(j)] = params['dW']
        grads['db' + str(j)] = params['db']
    return grads

def Update_parameters(parameters, grads, learning_rate):
    num_of_layers = parameters['layers_dim']
    for i in range(1, num_of_layers + 1):
        parameters['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
        parameters['b' + str(i)] -= learning_rate * grads['db' + str(i)]
    return parameters

'''
X: (height*width, number_of_examples)
Y: (num_of_class, number_of_examples)
'''
def L_layer_model(X, Y, layer_dims, learning_rate, num_iterations, batch_size):
    global UseDropout
    global UseBatchnorm
    global OnTrain
    costs = []
    costTrainStep = []
    bestValAccList = []
    parameters = initialize_parameters(layer_dims)
    x_train, x_val, y_train, y_val = split_train_validation(X, Y)
    num_of_examples = x_train.shape[1]

    bestValAcc = 0
    epoch = 0
    iteration = 0
    lastIterationWithChange = 0
    start = datetime.now()
    running = True
    while running:
        for j in range(0, num_of_examples, batch_size):
            OnTrain = True
            x_batch = x_train[:, j: j+batch_size]
            y_batch = y_train[:, j: j+batch_size]
            parameters['layers_dim'] = len(layer_dims)-1
            AL, caches = L_model_forward(x_batch,parameters, UseBatchnorm)
            parameters.update(caches)
            parameters['size_of_batch'] = batch_size
            loss = compute_cost(AL, y_batch)
            parameters['A0'] = x_batch
            grads = L_model_backward(AL, y_batch, parameters)
            parameters['layers_dim'] = len(layer_dims)-1
            parameters = Update_parameters(parameters, grads, learning_rate)

            if iteration % 100 == 0:
                costs.append(loss)
                costTrainStep.append(iteration)

            # if iteration%10 == 0:
            acc = Predict(x_val, y_val, parameters)
            if acc > bestValAcc + 0.0000001:
                lastIterationWithChange = iteration
                bestValAcc = acc
                bestValAccList.append(bestValAcc)

            if iteration - lastIterationWithChange > 600:
                running = False
                print('Done because of no improvement')
                break

            iteration += 1
            if iteration == num_iterations:
                running = False
                print('Done because of max iterations')

        if epoch % 5 == 0:
            print(f'loss at epoch {epoch + 1}, iteration: {iteration}:\t{loss}')

        epoch += 1
    elapsed = datetime.now() - start
    print('Time elapsed (hh:mm:ss.ms) {}'.format(elapsed))
    print(f'Train accuracy: {Predict(x_train, y_train, parameters)}')
    print(f'Validation accuracy: {Predict(x_val, y_val, parameters)}')
    plot(costs, costTrainStep)
    plot(bestValAccList)
    return parameters, costs

''' X: (height*width, number_of_examples)
    Y: (num_of_classes, number of examples)'''
def Predict(X, Y, parameters):
    global OnTrain
    OnTrain = False
    global UseBatchnorm
    AL, caches = L_model_forward(X, parameters, UseBatchnorm)
    AL = np.argmax(AL, axis=0)
    correct = Y[AL, np.indices(AL.shape)[0]].sum()
    total = AL.shape[0]
    return correct/total

def plot(costs, costsIdx=None):
    if costsIdx != None:
        plt.plot(costsIdx, costs)
    else:
        plt.plot(range(1, len(costs) + 1), costs)
    plt.show()

def main():
    print('Starting...')
    global UseBatchnorm
    UseBatchnorm = True
    global UseDropout
    UseDropout = True
    learning_rate = 0.009
    layer_dims = [784, 20, 7, 5, 10]
    iterations = 25_000
    batch_size = 150
    x_train, x_test, y_train, y_test = load_mnist()
    parameters, costs = L_layer_model(x_train, y_train, layer_dims, learning_rate, iterations, batch_size)
    accuracy = Predict(x_test, y_test, parameters)
    print(f'Test accuracy: {accuracy}')

if __name__ == '__main__':
    main()
