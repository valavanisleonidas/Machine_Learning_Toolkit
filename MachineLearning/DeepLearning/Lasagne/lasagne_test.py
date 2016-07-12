# NOTE :
# if you encounter problem with cnn
# update lasagne using command:
# sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt

from __future__ import print_function
import sys
sys.path.append('../../../')
import time
import numpy as np
import theano
import theano.tensor as T
import Utils.Matrix as matrix
import lasagne

theano.config.compute_test_value = 'off'


def mlp(input_var, shape, output_classes, num_units):
    # MLP of two hidden layers of default value 100 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    network = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)

    # Apply 20% dropout to the input data:
    network = lasagne.layers.DropoutLayer(network, p=0.2)

    # Add a fully-connected layer of num_units units, using the linear rectifier, and
    # initializing weights with GlorotUniform scheme (which is the default)
    # GlorotUniform: Glorot with weights sampled from the Uniform distribution.

    network = lasagne.layers.DenseLayer(
        network, num_units=num_units,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    # Another 800-unit layer:
    network = lasagne.layers.DenseLayer(
        network, num_units=num_units,
        nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    network = lasagne.layers.DropoutLayer(network, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    network = lasagne.layers.DenseLayer(
        network, num_units=output_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def custom_mlp(input_var, shape, output_classes, depth=2, width=100, drop_input=.2,
               drop_hidden=.5):
    # MLP which can be customized with respect to the number and size of hidden layers.


    # Input layer and DropoutLayer
    network = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(
            network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    network = lasagne.layers.DenseLayer(network,
                                        output_classes,
                                        nonlinearity=lasagne.nonlinearities.softmax)
    return network


def cnn2D(input_var, shape, output_classes, num_filter=32, filter_size=(5,5), pool_size=(128, 128)):
    # CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer
    network = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    print(lasagne.layers.get_output_shape(network))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(network,
                  num_filters=num_filter, filter_size=filter_size,
                  nonlinearity=lasagne.nonlinearities.rectify
                  ,pad="valid")

    print(lasagne.layers.get_output_shape(network))

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=pool_size)

    print(lasagne.layers.get_output_shape(network))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=output_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network

def cnn1D(input_var, shape, output_classes, num_filter=32, filter_size=10, pool_size=100):
    # CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer
    network = lasagne.layers.InputLayer(shape=shape,
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    print(lasagne.layers.get_output_shape(network))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv1DLayer(network,
                  num_filters=num_filter, filter_size=filter_size,
                  nonlinearity=lasagne.nonlinearities.rectify
                  ,pad="valid")

    print(lasagne.layers.get_output_shape(network))

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=pool_size)


    print(lasagne.layers.get_output_shape(network))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=256,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=output_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        # get indexes of inputs
        indices = np.arange(len(inputs))
        # shuffle indexes
        np.random.shuffle(indices)

    for index in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[index:index + batchsize]
        else:
            excerpt = slice(index, index + batchsize)
        yield inputs[excerpt], targets[excerpt]


def loadDataset():
    # Load the dataset
    print("Loading data...")

    trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy'
    testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy - Copy'


    #
    # trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TrainSet'
    # testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TestSet'

    # trainFolder = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\\data-CIELab-2\\train'
    # testFolder = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\\data-CIELab-2\\test'


    return matrix.load_dataset(trainFolder=trainFolder, testFolder=testFolder, imageSize=(100, 100))

def loadFeatures():
    print("Loading features...")

    trainPath = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\Features\\train_2x2_CIELab_512.txt'
    testPath = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\Features\\test_2x2_CIELab_512.txt'
    trainLabelPath = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\Features\\train_2x2_CIELab_512_labels.txt'
    testLabelPath = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\Features\\test_2x2_CIELab_512_labels.txt'



    trainPath = 'C:\Users\l.valavanis\Desktop\Clef2013\sentiment_embeddings\stage1\\features_train_subj'
    testPath = 'C:\Users\l.valavanis\Desktop\Clef2013\sentiment_embeddings\stage1\\features_test_subj'
    trainLabelPath = 'C:\Users\l.valavanis\Desktop\Clef2013\sentiment_embeddings\stage1\\labels_train_subj'
    testLabelPath = 'C:\Users\l.valavanis\Desktop\Clef2013\sentiment_embeddings\stage1\\labels_test_subj'



    return matrix.loadFeatures(trainPath=trainPath, trainLabels=trainLabelPath, testPath=testPath,
                                   testLabels=testLabelPath);


# NOTE!!!! :Output Classes Start from zero (0)
def main(model='mlp', num_epochs=3, num_batches=500):

    # use if you have an image dataset in 2D or 3D array
    # [trainArray, train_labels, testArray, test_labels, validationArray, validation_labels, outputClasses]=loadDataset()

    # use if you have features of images in 1D array
    [trainArray, train_labels, testArray, test_labels, validationArray, validation_labels,outputClasses] = loadFeatures()


    # Prepare Theano variables for inputs and targets
    target_var = T.ivector('targets')
    if (trainArray.ndim == 4):
        input_var = T.tensor4('inputs')
        # shape is a 4D array : num_batches,num_channels,rows,columns)
        shape = (None, trainArray.shape[1], trainArray.shape[2], trainArray.shape[3])
    else:
       # shape is a 3D array : (batch_size, num_input_channels, input_length)
       shape = (None, trainArray.shape[1], trainArray.shape[2])
       input_var = T.tensor3('inputs')

    print (validation_labels)
    print(outputClasses)
    # print (trainArray)
    print(trainArray.shape)
    print(train_labels.shape)
    print(testArray.shape)
    print(test_labels.shape)
    print(validationArray.shape)
    print(validation_labels.shape)

    model = "cnn"
    # Create neural network model (depending on first command line parameter)
    print("Building model: " + model + " and compiling functions...")
    if model == 'mlp':

        network = mlp(input_var=input_var, shape=shape, num_units=int(0.66 * train_labels.shape[0]),
                      output_classes=outputClasses)
    elif model == 'custom_mlp':
        depth = 50
        # 2/3 of training data
        width = int(0.66 * train_labels.shape[0])
        drop_in = .2
        drop_hid = .5


        network = custom_mlp(input_var=input_var, shape=shape, depth=depth
                             , width=width, drop_input=drop_in
                             , drop_hidden=drop_hid, output_classes=outputClasses)
    elif model == 'cnn':
        # if images are given ( 4D arrays) implement CNN with 2D convolutions
        if (trainArray.ndim == 4):
             network = cnn2D(input_var=input_var, shape=shape, output_classes=outputClasses)
        else:
            network = cnn1D(input_var=input_var, shape=shape, output_classes=outputClasses)

    else:
        print("Unrecognized model type %r." % model)
        return

    print("creating loss functions and update parameters")

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training, how to modify the
    # parameters at each training step
    params = lasagne.layers.get_all_params(network, trainable=True)

    # updates = lasagne.updates.adadelta(loss, params, learning_rate=0.01)
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    updates = lasagne.updates.adagrad(loss_or_grads=loss, params=params, learning_rate=0.001)

    # Create a loss expression for validation/testing.
    # we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # predict values
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(trainArray, train_labels, num_batches, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(validationArray, validation_labels, num_batches, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f} ".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(testArray, test_labels, num_batches, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f} ".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # get predictions for array
    output_Values = predict_fn(testArray)
    print(np.array(output_Values))

    # Optionally, you could now save the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    main()
