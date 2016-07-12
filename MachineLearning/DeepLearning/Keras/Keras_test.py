from __future__ import print_function
import sys
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, TimeDistributedDense, RepeatVector, Merge
from keras.preprocessing.image import ImageDataGenerator

sys.path.append('../../../')

from Utils import Matrix as matrix
from Utils import Load as load
import numpy as np
from keras.initializations import normal, identity
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils, generic_utils
import theano

theano.config.warn_float64 = 'ignore'


# REALLY FASTEEEEEEEEEEEEEEEEEER
# theano.config.floatX = 'float32'
# theano.sandbox.cuda.use("gpu0")
# theano.config.allow_gc = False

# THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu,allow_gc=False,cuda.root=/usr/local/cuda-7.5 python Keras_test.py



# TODO TESTIING:
#
#
#
# TODO CHECK TRAIN , train_MINIBATCHES
# TODO TEST WITH MANY CHANGED PARAMETERS ,,, experiments with other colors other than RGB ,,,
# TODO TEST MLP. SEEMS GOOD ENOUGH SO FAR
#
# TODO IMPORTAAAAAAAAAAAAAAAAANT
# TODO FIND HOW TO TUNE SVM WITH MANY CLASSIFIERS



# TODO BUGS PROBLEM:
#
# TODO Problem with float 32 something is FLOAT 64 : SOLVED IT WAS DROPOUUUUUUUUUUUT BUT WE WOULD LIKE TO HAVE THAT AS A PARAMETER



# TODO FEATURES IMPLEMENTATION:
#
#
#
# TODO learning curves: cnn with minibatches.
# TODO FIND GOOD CNN MODEL
# TODO pretrained models
# TODO Extract features from CNN trained model
# TODO FIND WHICH COLOR SPACE SUITS AS BEST
# TODO thourough TUNING OR NOT
# TODO tuning with image size convertion , image generator , normalize range , preprocess iamge ktlp
# TODO BAYES OPTIMIZATION



# TODO MEMORY IN VALIDATION
# TODO load only TRAIN and then TEST
# TODO check minibatches

# TODO COMPLETED :
#
# TODO CHECK XYZ COLOR SPACE
# TODO Convert 0,1 also HSV AND CIELAB . with / 255 only RGB is within 0,1
#

class CNN:
    # The general formula for calculating output size based on filter parameters is (D+2P-F)/S+1 where D is the input
    # size of one dimension (assuming the input is a square), P is the padding, F is the filter size, and S is the stride.


    def __init__(self):
        pass

    def train(self, features, labels,val_features=None,val_labels=None, outputClasses=None, batch_size=128, nb_epoch=50, nb_filters=10, nb_pool=2,
              nb_conv=2, num_inputs=1024, depth=1, border_mode='same', activation='relu', optimizer='sgd',
              data_augmentation=True, learning_curves_OR_Cross_Val=False):

        # if train is not for creating learning curves all labels
        # will be given so extract number of labels by array
        if not learning_curves_OR_Cross_Val:
            assert val_features != None, "Give validation features"
            assert val_labels != None, "Give validation labels"

            outputClasses = matrix.Matrix().getNumberOfClasses(labels)

        assert outputClasses != None, "Give Number of classes"
        # assert outputClasses==matrix.Matrix().getNumberOfClasses(labels),"Number of classes given should match the labels array!"

        print('X_train shape:', features.shape)
        print(features.shape[0], 'train samples')

        # train on batches
        if data_augmentation:
            if learning_curves_OR_Cross_Val:
                [features, val_features, labels, val_labels] = matrix.Matrix().SplitTrainValidation(
                    trainArray=features, train_labels=labels,takeLastExamples=True,maxImagesPerCategory=int(.2*features.shape[0]/outputClasses))

            ValCategorical_labels = self.convertLabelsToCategorical(val_labels, outputClasses)


        Categorical_labels = self.convertLabelsToCategorical(labels, outputClasses)

        print('X_train shape:', features.shape)
        print(features.shape[0], 'train samples')
        print(labels.shape, 'labels shape')
        print('val shape:', val_features.shape)
        print(val_features.shape[0], 'val samples')
        print(val_labels.shape, 'val labels shape')
        print("creating model!")
        # create model architecture
        # model = self.modelArchitecture(features=features,outputClasses=outputClasses,nb_filters=features.shape[2],nb_pool=nb_pool,nb_conv=nb_conv)
        model = self.modelArchitecture_Improved(channels=features.shape[1], rows=features.shape[2],
                                                columns=features.shape[3], outputClasses=outputClasses,
                                                nb_filters=nb_filters, nb_pool=nb_pool, nb_conv=nb_conv,
                                                num_inputs=num_inputs, depth=depth, border_mode=border_mode,
                                                activation=activation, optimizer=optimizer)
        # model = self.modelArchitecture_Improved2(features=features,outputClasses=outputClasses,nb_filters=features.shape[2],nb_pool=nb_pool,nb_conv=nb_conv)
        print("finished model!")

        early_stopping = EarlyStopping(monitor='val_loss', patience=6)
        if not data_augmentation:
            print('Not using data augmentation.')

            model.fit(features, Categorical_labels, batch_size=batch_size, nb_epoch=nb_epoch,
                      show_accuracy=True, verbose=1 ,validation_data=(val_features, ValCategorical_labels),
                      callbacks=[early_stopping])

        else:
            print('Using real-time data augmentation.')

            # create Image Generator
            datagen = self.createImageGenerator()
            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(features)

            # fit the model on the batches generated by datagen.flow()
            model.fit_generator(datagen.flow(features, Categorical_labels, batch_size=batch_size),
                                samples_per_epoch=features.shape[0],
                                nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
                                validation_data=(val_features, ValCategorical_labels),
                                nb_worker=4, callbacks=[early_stopping])

            for e in range(nb_epoch):
                print('Epoch %d/%d  ' % (e + 1, nb_epoch))
                batches = 0
                progbar = generic_utils.Progbar(features.shape[0])

                for X_batch, Y_batch in datagen.flow(features, Categorical_labels, batch_size=batch_size):
                    loss = model.train_on_batch(X_batch, Y_batch, accuracy=True)
                    progbar.add(X_batch.shape[0])
                    print(" - train loss: %.4f - train acc: %.4f " % (loss[0], loss[1]))
                    batches += 1
                    if batches > len(features) / batch_size:
                        # we need to break the loop by hand because
                        # the generator loops indefinitely
                        break

        return model

    def train_with_MiniBatches(self, trainFolder, outputClasses=None, image_size=(256, 256), image_channels=3,
                               batch_size=60, nb_epoch=10, nb_pool=2, nb_conv=3, data_augmentation=True):

        assert outputClasses != None, "Give Number of classes"

        print("Creating model Architecture...")
        # create model architecture
        # model = self.modelArchitecture(features=features,outputClasses=outputClasses,nb_filters=features.shape[2],nb_pool=nb_pool,nb_conv=nb_conv)
        model = self.modelArchitecture_Improved(channels=image_channels, rows=image_size[0], columns=image_size[1],
                                                outputClasses=outputClasses, nb_filters=image_size[1],
                                                nb_pool=nb_pool, nb_conv=nb_conv)
        # model = self.modelArchitecture_Improved2(features=features,outputClasses=outputClasses,nb_filters=features.shape[2],nb_pool=nb_pool,nb_conv=nb_conv)

        # create Image Generator
        datagen = self.createImageGenerator()

        # for every minibatch read images and train on batches
        for features, labels in load.load_dataset_minibatches(folderPath=trainFolder, image_size=image_size,
                                                              batch_size=batch_size):

            print('Train batch shape:', features.shape)

            if data_augmentation:
                [features, val_features, labels, val_labels] = matrix.Matrix().SplitTrainValidation(
                    trainArray=features, train_labels=labels,takeLastExamples=False,maxImagesPerCategory=int(.2*features.shape[0]/outputClasses))


                # reshape array for categorical cross entropy
                ValCategorical_labels = self.convertLabelsToCategorical(val_labels, outputClasses)

            Categorical_labels = self.convertLabelsToCategorical(labels, outputClasses)

            print(labels.shape, 'labels shape')

            early_stopping = EarlyStopping(monitor='val_loss', patience=6)

            if not data_augmentation:

                print('Not using data augmentation.')
                for e in range(nb_epoch):
                    print('Epoch %d/%d  ' % (e + 1, nb_epoch))
                    batches = 0
                    progbar = generic_utils.Progbar(features.shape[0])

                    for X_batch, Y_batch in datagen.flow(features, Categorical_labels, batch_size=batch_size):
                        loss = model.train_on_batch(X_batch, Y_batch, accuracy=True)
                        progbar.add(features.shape[0])
                        print(" - train loss: %.4f - train acc: %.4f " % (loss[0], loss[1]))
                        batches += 1
                        if batches > len(features) / batch_size:
                            # we need to break the loop by hand because
                            # the generator loops indefinitely
                            break

            else:
                print('Using real-time data augmentation.')
                # compute quantities required for featurewise normalization
                # (std, mean, and principal components if ZCA whitening is applied)
                datagen.fit(features)

                # fit the model on the batches generated by datagen.flow()
                model.fit_generator(datagen.flow(features, Categorical_labels, batch_size=batch_size),
                                    samples_per_epoch=features.shape[0],
                                    nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
                                    validation_data=(val_features, ValCategorical_labels),
                                    nb_worker=4, callbacks=[early_stopping])

                for e in range(nb_epoch):
                    print('Epoch %d/%d  ' % (e + 1, nb_epoch))
                    batches = 0
                    progbar = generic_utils.Progbar(features.shape[0])

                    for X_batch, Y_batch in datagen.flow(features, Categorical_labels, batch_size=batch_size):
                        print (batches , ' // ', len(features) / batch_size)
                        loss = model.train_on_batch(X_batch, Y_batch, accuracy=True)
                        progbar.add(features.shape[0])
                        print(" - train loss: %.4f - train acc: %.4f " % (loss[0], loss[1]))
                        batches += 1
                        if batches > len(features) / batch_size:
                            # we need to break the loop by hand because
                            # the generator loops indefinitely
                            break

        return model

    def predict(self, features, model, labels, learning_curves_OR_Cross_Val=False, outputClasses=None, batch_size=128):

        if not learning_curves_OR_Cross_Val:
            outputClasses = len(np.unique(labels))

        assert outputClasses != None, "Give Number of classes"

        Categorical_labels = self.convertLabelsToCategorical(labels, outputClasses)

        print('\n', features.shape[0], 'test samples')

        score = model.evaluate(features, Categorical_labels, batch_size=batch_size, show_accuracy=True, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score[1]

    def predictClasses(self, features, model, batch_size=128):
        return model.predict_classes(features, batch_size=batch_size)

    def predictProba(self, features, model, batch_size=128):
        return model.predict_proba(features, batch_size=batch_size)

    def predict_with_MiniBatches(self, testFoler, model, outputClasses=None, image_size=(256, 256), batch_size=50):
        assert outputClasses != None, "Give Number of classes"

        datagen = self.createImageGenerator()
        acc = []
        for features, labels in load.load_dataset_minibatches(folderPath=testFoler,
                                                              image_size=image_size, batch_size=batch_size):

            print('test batch shape:', features.shape)
            print(features.shape[0], 'test samples')

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(features)

            Categorical_labels = self.convertLabelsToCategorical(labels, outputClasses)

            print('\n', features.shape[0], 'test samples')

            batches = 0
            progbar = generic_utils.Progbar(features.shape[0])
            for X_batch, Y_batch in datagen.flow(features, Categorical_labels, batch_size=batch_size):
                score = model.test_on_batch(X_batch, Y_batch, accuracy=True)
                acc.append(score[1])
                progbar.add(features.shape[0])
                print(" - test loss: %.4f - test acc: %.4f " % (score[0], score[1]))
                batches += 1
                if batches > len(features) / batch_size:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break

            score = model.evaluate(features, Categorical_labels, batch_size=batch_size, show_accuracy=True, verbose=0)
            print('\nTest score:', score[0])
            print('Test accuracy:', score[1])

        print('Final accuracy:', np.mean(acc))
        return acc

    def createImageGenerator(self, featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             rotation_range=0,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=False):
        # this will do preprocessing and realtime data augmentation
        return ImageDataGenerator(
            featurewise_center=featurewise_center,  # set input mean to 0 over the dataset
            samplewise_center=samplewise_center,  # set each sample mean to 0
            featurewise_std_normalization=featurewise_std_normalization,  # divide inputs by std of the dataset
            samplewise_std_normalization=samplewise_std_normalization,  # divide each input by its std
            zca_whitening=zca_whitening,  # apply ZCA whitening
            rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=height_shift_range,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=horizontal_flip,  # randomly flip images
            vertical_flip=vertical_flip)  # randomly flip images

    def convertLabelsToCategorical(self, labels, outputClasses):
        labels = np.reshape(labels, (len(labels), 1))
        return np_utils.to_categorical(labels, outputClasses)

    # better of all so far
    def modelArchitecture_Improved(self, channels, rows, columns, outputClasses, nb_filters=32, nb_pool=2, nb_conv=3
                                   , border_mode='same', activation='relu', num_inputs=512, depth=1, optimizer='sgd'):

        model = Sequential()

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode=border_mode,
                                input_shape=(channels, rows, columns)))
        model.add(Activation(activation))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        for _ in range(0, depth):
            print('Hidden Layers')
            model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv, border_mode=border_mode))
            model.add(Activation(activation))
            model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv))
            model.add(Activation(activation))
            model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
            model.add(Dropout(0.25))

        assert (model.output_shape[2] != 0 and model.output_shape[3] != 0), \
            'Output size cannot be zero (shape must be at least (1,1) ). Change architecture to decrease convolution,' \
            ' pooling operations or increase size of shape'

        model.add(Flatten())
        model.add(Dense(num_inputs))
        model.add(Activation(activation))
        model.add(Dropout(0.5))
        model.add(Dense(outputClasses))
        model.add(Activation('softmax'))

        # optimizers
        if optimizer == 'adam':
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        elif optimizer == 'sgd':
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd)
        elif optimizer == 'rms':
            rms = RMSprop()
            model.compile(loss='categorical_crossentropy', optimizer=rms)
        elif optimizer == 'adadelta':
            model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        elif optimizer == 'adagrad':
            model.compile(loss='categorical_crossentropy', optimizer='adagrad')

        return model

    # second better
    def modelArchitecture_Improved2(self, features, outputClasses, nb_filters, nb_pool, nb_conv):

        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(features.shape[1], features.shape[2], features.shape[3])))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        assert (model.output_shape[2] != 0 and model.output_shape[3] != 0), \
            'Output size cannot be zero (shape must be at least (1,1) ). Change architecture to decrease convolution,' \
            ' pooling operations or increase size of shape'

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(outputClasses, activation='softmax'))

        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        return model

    # worse
    def modelArchitecture(self, features, outputClasses, nb_filters, nb_pool, nb_conv):
        model = Sequential()

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                border_mode='valid',
                                input_shape=(features.shape[1], features.shape[2], features.shape[3])))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Activation('relu'))

        model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.25))

        assert (model.output_shape[2] != 0 and model.output_shape[3] != 0), \
            'Output size cannot be zero (shape must be at least (1,1) ). Change architecture to decrease convolution,' \
            ' pooling operations or increase size of shape'

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(outputClasses))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        return model

    # Architecture for learning image captions with a convnet and a Gated Recurrent Unit
    def modelArchitecture_CnnWithGRU(self, features, outputClasses, nb_filters, nb_pool=2, nb_conv=3):
        max_caption_len = 16
        vocab_size = 10000

        # first, let's define an image model that
        # will encode pictures into 128-dimensional vectors.
        # it should be initialized with pre-trained weights.
        image_model = Sequential()
        image_model.add(Convolution2D(32, nb_conv, nb_conv, border_mode='valid',
                                      input_shape=(features[1], features[2], features[3])))
        image_model.add(Activation('relu'))
        image_model.add(Convolution2D(32, nb_conv, nb_conv))
        image_model.add(Activation('relu'))
        image_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

        image_model.add(Convolution2D(64, nb_conv, nb_conv, border_mode='valid'))
        image_model.add(Activation('relu'))
        image_model.add(Convolution2D(64, nb_conv, nb_conv))
        image_model.add(Activation('relu'))
        image_model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

        image_model.add(Flatten())
        image_model.add(Dense(128))

        # let's load the weights from a save file.
        image_model.load_weights('weight_file.h5')

        # next, let's define a RNN model that encodes sequences of words
        # into sequences of 128-dimensional word vectors.
        language_model = Sequential()
        language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
        language_model.add(GRU(output_dim=128, return_sequences=True))
        language_model.add(TimeDistributedDense(128))

        # let's repeat the image vector to turn it into a sequence.
        image_model.add(RepeatVector(max_caption_len))

        # the output of both models will be tensors of shape (samples, max_caption_len, 128).
        # let's concatenate these 2 vector sequences.
        model = Merge([image_model, language_model], mode='concat', concat_axis=-1)
        # let's encode this vector sequence into a single vector
        model.add(GRU(256, 256, return_sequences=False))
        # which will be used to compute a probability
        # distribution over what the next word in the caption should be!
        model.add(Dense(vocab_size))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        # "images" is a numpy float array of shape (nb_samples, nb_channels=3, width, height).
        # "captions" is a numpy integer array of shape (nb_samples, max_caption_len)
        # containing word index sequences representing partial captions.
        # "next_words" is a numpy float array of shape (nb_samples, vocab_size)
        # containing a categorical encoding (0s and 1s) of the next word in the corresponding
        # partial caption.


        # model.fit([images, partial_captions], next_words, batch_size=16, nb_epoch=100)

        return model


def cnn(train=False,
        learning_curves=True,
        train_minibatches=False,
        tuning=False):
    print("CNN")

    # trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SetWithValidation\TrainSet'
    # testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TestSet'

    # trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2012\SampleImages - Copy'
    # testFolder = 'C:\Users\l.valavanis\Desktop\Clef2012\SampleImages - Copy - Copy'

    # trainFolder = '/home/leonidas/Documents/ImageClef2016/Clef2013/TrainSet'
    # testFolder = '/home/leonidas/Documents/ImageClef2016/Clef2013/TestSet'

    # trainFolder = '/home/leonidas/Desktop/images/SampleImages - Copy - Copy'
    # testFolder = '/home/leonidas/Desktop/images/SampleImages - Copy'
    #
    trainFolder = '/home/leonidas/Desktop/images/TrainSet'
    testFolder = '/home/leonidas/Desktop/images/TestSet'

    # Image Settings
    imageSize = (40, 40)
    convertion = 'RGB'
    image_channels = 3
    normalize = True
    normalizeRange = (-1, 1)

    # tuning + parameters
    classifier = 'CNN'
    performCrossVal = False
    splitToValidationSet=True
    maxImagesPerCategory=10


    batch_size = 128
    depth = 1
    nb_epoch = 30
    # print(convertion, ' : ', normalizeRange, ' : epochs :', nb_epoch)

    nb_filter = 32
    # nb_filters=trainArray.shape[2]
    nb_pool = 2
    nb_conv = 3
    activation = 'relu'
    optimizer = 'sgd'
    num_inputs = 1024
    border_mode = 'same'

    if train_minibatches:
        batch_size = 200
        print("minibatches")
        outputClasses = 30

        # TRAIN WITH MINIBATCHES
        model = CNN().train_with_MiniBatches(trainFolder=trainFolder, image_size=imageSize,
                                             image_channels=image_channels, data_augmentation=True
                                             , outputClasses=outputClasses, nb_epoch=nb_epoch, batch_size=batch_size)

        predictedLabels = CNN().predict_with_MiniBatches(testFoler=testFolder, image_size=imageSize,
                                                         model=model, outputClasses=outputClasses,
                                                         batch_size=batch_size)

    if learning_curves:
        print("curves")

        from MachineLearning import learningCurves
        import platform

        ImagePixelSizes = [(20, 20)  ]
        epoches = ([100 , 200 , 300  ])
        ImageNormalizeRanges = [(0, 1), (-1, 1)]
        ImagecolorSpaces = [ 'RGB', 'HSV', 'Grayscale', 'CieLab' , 'XYZ' , 'RGBCIE' , 'LUV' ]

        ImagePixelSizes = [(40, 40) ]
        epoches = ([  200,300 , 500  ])
        ImageNormalizeRanges = [ (-1, 1) , (0, 1)]
        ImagecolorSpaces = [ 'RGB'  ]

        # CLEF 2012 :
        # ImagePixelSizes = [ (10 , 10 ) (20, 20) , (30, 30) , (40, 40) ]
        # epoches = ([10 , 50 ,100 , 200  ])
        # ImageNormalizeRanges = [(0, 1), (-1, 1)]
        # ImagecolorSpaces = [ 'RGB', 'HSV', 'Grayscale', 'CieLab' , 'XYZ' , 'RGBCIE' , 'LUV' ]

        for colorSpace in ImagecolorSpaces:
            for imageSize in ImagePixelSizes:
                for normRange in ImageNormalizeRanges:
                    print ('loading ' , colorSpace , ' ' , imageSize,' in range values ', normRange)

                    [trainArray, train_labels, testArray, test_labels, outputClasses] = \
                        load.load_dataset(trainFolder, testFolder,imageSize=imageSize, convertion=convertion,
                        normalize=normalize, normalizeRange=normalizeRange, imageChannels=image_channels,
                        splitToValidationSet=False,maxImagesPerCategory=maxImagesPerCategory)

                    for epoch in epoches:

                        imageName = '{0}_imSize_{8}x{8}_{1} epochs_{2}_{3} norm_{4}_{5} pool_{6} conv_{7}.png'.format(colorSpace, epoch,
                                      activation, optimizer,normRange[0],normRange[1],
                                      nb_pool, nb_conv,imageSize[0])

                        savePathUbuntu = '/home/leonidas/Dropbox/clef2016/experiments/Clef2013/Learning Curves/{0}'.format(imageName)
                        savePathWindows = 'C:\Users\l.valavanis\Dropbox\clef2016\experiments\Clef2013\Learning Curves\{0}'.format(imageName)
                        savePath = savePathUbuntu
                        print('save path : ', savePath)
                        if platform.system() == "Windows":
                            savePath = savePathWindows
                        else:
                            savePath = savePathUbuntu

                        import time
                        start_time = time.time()

                        learningCurves.plot_learning_curve(features_train=trainArray, labels_train=train_labels,
                                                           features_test=testArray,labels_test=test_labels,
                                                           outputClasses=outputClasses,classifier=classifier,
                                                           showFigure=False, saveFigure=True, savePath=savePath,
                                                           nb_epochs=epoch)
                        elapsed_time = time.time() - start_time
                        print(elapsed_time, ' seconds')




    if learning_curves !=True:
        [trainArray, train_labels, testArray, test_labels, val_features, val_labels, outputClasses] = \
                load.load_dataset(trainFolder, testFolder,imageSize=imageSize, convertion=convertion,
                normalize=normalize, normalizeRange=normalizeRange, imageChannels=image_channels,
                                  takeLastExamples=True,
                splitToValidationSet=splitToValidationSet,maxImagesPerCategory=maxImagesPerCategory)


    if train:

        print (val_features.shape)
        print (val_labels.shape)
        print("train")
        import time
        start_time = time.time()

        model = CNN().train(features=trainArray, labels=train_labels,val_features=val_features,val_labels=val_labels,

                            outputClasses=outputClasses,
                            nb_epoch=nb_epoch,
                            nb_filters=trainArray.shape[2],
                            nb_pool=nb_pool,
                            nb_conv=nb_conv,
                            activation=activation,
                            optimizer=optimizer,
                            num_inputs=num_inputs,
                            depth=depth,
                            border_mode=border_mode,
                            data_augmentation=True)
        predictedLabels = CNN().predict(features=testArray, labels=test_labels, model=model)
        elapsed_time = time.time() - start_time
        print(elapsed_time, ' seconds')




    if tuning:
        [trainArray, train_labels, testArray, test_labels, outputClasses] = \
                load.load_dataset(trainFolder, testFolder,imageSize=imageSize, convertion=convertion,
                normalize=normalize, normalizeRange=normalizeRange, imageChannels=image_channels,
                splitToValidationSet=False,maxImagesPerCategory=maxImagesPerCategory)

        from Utils.Tuning.NeuralNetworks_Tuning import neural_tuning

        print("Tuning")
        import time
        start_time = time.time()

        [bestAccuracy, bestBatchSize, bestEpochs, bestNbFilters, bestNbPool, bestNbConv, bestBorderMode, bestActivation,
         bestNumInputs, bestDepth, bestOptimizer] \
            = neural_tuning(features_train=trainArray, labels_train=train_labels, features_test=testArray,
                            labels_test=test_labels,
                            performCrossVal=performCrossVal,
                            outputClasses=outputClasses,
                            modelType="CNN",
                            batch_size=(10, 100, 50),
                            nb_epoch=(10, 100, 20),
                            nb_filters=(10, 50, 5),
                            nb_pool=(2, 5, 1),
                            nb_conv=(2, 5, 1),
                            border_mode=['same', 'valid'],
                            activation=['relu', 'tanh'],
                            num_input=(100, 1024, 100),
                            depths=(1, 2, 1),
                            optimizers=['adam', 'sgd', 'rms', 'adadelta', 'adagrad'], )
        print('image size:', imageSize)
        print('best acc:', bestAccuracy)
        print('best batch size:', bestBatchSize)
        print('best epochs:', bestEpochs)
        print('best filter:', bestNbFilters)
        print('best pool size:', bestNbPool)
        print('best conv:', bestNbConv)
        print('best activ :', bestActivation)
        print('best optimizer :', bestOptimizer)
        print('best inputs :', bestNumInputs)
        print('best depth :', bestDepth)
        print('best border :', bestBorderMode)
        elapsed_time = time.time() - start_time
        print(elapsed_time, ' seconds')

    print(convertion, ' : ', normalizeRange, ' : ', imageSize)


class MLP:
    def __init__(self):
        pass

    def convertLabelsToCategorical(self, labels, outputClasses):
        labels = np.reshape(labels, (len(labels), 1))
        return np_utils.to_categorical(labels, outputClasses)

    def train(self, features, labels, outputClasses=None, batch_size=500, nb_epoch=200,
              activation='relu', num_input=1024, depth=1, optimizer='sgd', learning_curves_OR_Cross_Val=False):

        if not learning_curves_OR_Cross_Val:
            outputClasses = len(np.unique(labels))
        assert outputClasses != None, "Give Number of classes"

        features = features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])

        Categorical_labels = self.convertLabelsToCategorical(labels, outputClasses)

        print('X_train shape:', features.shape)
        print(features.shape[0], 'train samples')

        print(labels.shape, 'labels shape')

        model = self.modelArchitecture(features=features, outputClasses=outputClasses,
                                       activation=activation, num_input=num_input, depth=depth, optimizer=optimizer)

        early_stopping = EarlyStopping(monitor='val_loss', patience=30)
        model.fit(features, Categorical_labels,
                  batch_size=batch_size, nb_epoch=nb_epoch,
                  show_accuracy=True, verbose=2,
                  validation_split=.25, callbacks=[early_stopping])

        return model

    def predict(self, features, model, labels=None, learning_curves_OR_Cross_Val=False, outputClasses=None,
                ShowAccuracy=True):

        if not learning_curves_OR_Cross_Val:
            outputClasses = len(np.unique(labels))

        assert outputClasses != None, "Give Number of classes"

        print(features.shape[0], 'test samples')

        features = features.reshape(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3])

        if ShowAccuracy:
            Categorical_labels = self.convertLabelsToCategorical(labels, outputClasses)

            print(features.shape[0], 'test samples')
            score = model.evaluate(features, Categorical_labels, show_accuracy=True, verbose=0)
            print('Test score:', score[0])
            print('Test accuracy:', score[1])
            return score[1]
        else:
            return model.predict_classes(features)

    def modelArchitecture(self, features, outputClasses, activation='relu', num_input=1024, depth=1, optimizer='sgd'):

        model = Sequential()
        model.add(Dense(0.66 * features.shape[0], input_shape=(features.shape[1],)))
        model.add(Activation(activation))
        model.add(Dropout(.2))

        for _ in range(depth):
            model.add(Dense(num_input))
            model.add(Activation(activation))
            model.add(Dropout(.5))

        model.add(Dense(num_input))
        model.add(Activation(activation))
        model.add(Dropout(.2))

        model.add(Dense(outputClasses))
        model.add(Activation('softmax'))

        # optimizers
        if optimizer == 'adam':
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        elif optimizer == 'sgd':
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd)
        elif optimizer == 'rms':
            rms = RMSprop()
            model.compile(loss='categorical_crossentropy', optimizer=rms)
        elif optimizer == 'adadelta':
            model.compile(loss='categorical_crossentropy', optimizer='adadelta')
        elif optimizer == 'adagrad':
            model.compile(loss='categorical_crossentropy', optimizer='adagrad')

        return model


def mlp(train=False, learning_curves=True, tuning=False):
    print("MLP")

    # trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy'
    # testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy - Copy'

    # trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TrainSet'
    # testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TestSet'

    # trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2012\TrainSet'
    # testFolder = 'C:\Users\l.valavanis\Desktop\Clef2012\TestSet'

    trainFolder = '/home/leonidas/Desktop/clef2016/Clef2013/TrainSet'
    testFolder = '/home/leonidas/Desktop/clef2016/Clef2013/TestSet'

    # trainFolder = '/home/leonidas/Desktop/images/SampleImages - Copy - Copy'
    # testFolder = '/home/leonidas/Desktop/images/SampleImages - Copy'

    activation = 'relu'
    optimizer = 'sgd'
    outputClasses = 30
    imageSize = (30, 30)
    convertion = "RGB"
    nb_epoch = 30
    classifier = "MLP"
    performCrossVal = True
    num_input=1024

    if learning_curves != True:
        [trainArray, train_labels, testArray, test_labels, nb_classes] = \
            load.load_dataset(trainFolder, testFolder, imageSize=imageSize, convertion=convertion)

    if train:
        print("Train")
        model = MLP().train(features=trainArray, labels=train_labels, nb_epoch=nb_epoch)
        accuracy = MLP().predict(features=testArray, labels=test_labels, model=model)

    if learning_curves:
        print("curves")

        from MachineLearning import learningCurves
        import platform

        ImagePixelSizes = [ (10,10) ,(20, 20) , (30, 30) , (40, 40) ]
        epoches = ([100 , 200 , 400 , 600 , 800 , 1000  ])
        ImageNormalizeRanges = [(0, 1), (-1, 1)]
        ImagecolorSpaces = [ 'RGB', 'HSV', 'Grayscale', 'CieLab' , 'XYZ' , 'RGBCIE' , 'LUV' ]



        for colorSpace in ImagecolorSpaces:
            for imageSize in ImagePixelSizes:
                for normRange in ImageNormalizeRanges:
                    print ('loading ' , colorSpace , ' ' , imageSize,' in range values ', normRange)

                    [trainArray, train_labels, testArray, test_labels, outputClasses] = \
                    load.load_dataset(trainFolder, testFolder, imageSize=imageSize, convertion=colorSpace,
                              normalize=True, normalizeRange=normRange, imageChannels=3)

                    for epoch in epoches:

                        imageName = '{0}_imSize_{6}x{6}_{1} epochs inputs_{7}_{2}_{3} norm_{4}_{5}.png'.format(colorSpace, epoch,
                                      activation, optimizer,normRange[0],normRange[1],imageSize[0],num_input)

                        savePathUbuntu = '/home/leonidas/Dropbox/clef2016/experiments/Clef2013/Learning Curves/{0}'.format(imageName)
                        savePathWindows = 'C:\Users\l.valavanis\Dropbox\clef2016\experiments\Clef2013\Learning Curves\{0}'.format(imageName)
                        savePath = savePathUbuntu
                        print('save path : ', savePath)
                        if platform.system() == "Windows":
                            savePath = savePathWindows
                        else:
                            savePath = savePathUbuntu

                        import time
                        start_time = time.time()

                        learningCurves.plot_learning_curve(features_train=trainArray, labels_train=train_labels,
                                                           features_test=testArray,labels_test=test_labels,
                                                           outputClasses=outputClasses,classifier=classifier,
                                                           showFigure=False, saveFigure=True, savePath=savePath,
                                                           nb_epochs=epoch)
                        elapsed_time = time.time() - start_time
                        print(elapsed_time, ' seconds')




    if tuning:
        from Utils.Tuning.NeuralNetworks_Tuning import neural_tuning

        print("Tuning")
        import time
        start_time = time.time()

        [bestAccuracy, bestBatchSize, bestEpochs, bestActivation, bestOptimizer, bestNumInputs,
         bestDepth] = neural_tuning(features_train=trainArray,
                                    labels_train=train_labels, features_test=testArray, labels_test=test_labels,
                                    performCrossVal=performCrossVal,
                                    outputClasses=nb_classes,
                                    modelType=classifier,
                                    batch_size=(10, 100, 50),
                                    nb_epoch=(10, 50, 30),
                                    activation=['relu', 'tanh'],
                                    num_input=(100, 200, 100),
                                    depths=(1, 2, 1),
                                    optimizers=['adam', 'sgd', 'rms', 'adadelta', 'adagrad'])
        print('image size:', imageSize)
        print('best acc:', bestAccuracy)
        print('best batch size:', bestBatchSize)
        print('best epochs:', bestEpochs)
        print('best activation:', bestActivation)
        print('best optimizer :', bestOptimizer)
        print('bestNumInputs:', bestNumInputs)
        print('bestDepth:', bestDepth)

        elapsed_time = time.time() - start_time
        print(elapsed_time, ' seconds')


class RNN:
    def __init__(self):
        pass

    def trainRNN(self, features, labels, outputClasses=None, batch_size=100, nb_epochs=10, hidden_units=1000,
                 learning_rate=1e-6, learning_curves=False):
        assert outputClasses != None, "Give Number of classes"

        if not learning_curves:
            outputClasses = len(np.unique(labels))

        features = features.reshape(features.shape[0], -1, features.shape[1])

        print('X_train shape:', features.shape)
        print(features.shape[0], 'train samples')

        # convert class vectors to binary class matrices

        Categorical_labels = self.convertLabelsToCategorical(labels, outputClasses)

        model = Sequential()
        model.add(SimpleRNN(output_dim=hidden_units,
                            init=lambda shape: normal(shape, scale=0.001),
                            inner_init=lambda shape: identity(shape, scale=1.0),
                            activation='relu', input_shape=features.shape[1:]))
        model.add(Dense(outputClasses))
        model.add(Activation('softmax'))

        rmsprop = RMSprop(lr=learning_rate)

        model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

        model.fit(features, labels, batch_size=batch_size, nb_epoch=nb_epochs,
                  show_accuracy=True, verbose=1, validation_split=.25)

        return model

    def trainRnnLSTM(self, features, labels, outputClasses=None, batch_size=100, nb_epochs=10, hidden_units=1000,
                     learning_rate=1e-6, learning_curves=False):
        assert outputClasses != None, "Give Number of classes"

        if not learning_curves:
            outputClasses = len(np.unique(labels))

        features = features.reshape(features.shape[0], -1, features.shape[1])

        print('X_train shape:', features.shape)
        print(features.shape[0], 'train samples')

        # convert class vectors to binary class matrices

        Categorical_labels = self.convertLabelsToCategorical(labels, outputClasses)

        model = Sequential()
        model.add(LSTM(hidden_units, input_shape=features.shape[1:]))
        model.add(Dense(outputClasses))
        model.add(Activation('softmax'))

        rmsprop = RMSprop(lr=learning_rate)

        model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

        model.fit(features, labels, batch_size=batch_size, nb_epoch=nb_epochs,
                  show_accuracy=True, verbose=1, validation_split=.25)

        return model

    def predict(self, features, model, labels=None, ShowAccuracy=True):
        print(features.shape[0], 'test samples')

        features = features.reshape(features.shape[0], -1, features.shape[1])

        if ShowAccuracy:
            Categorical_labels = self.convertLabelsToCategorical(labels, len(np.unique(labels)))

            print(features.shape[0], 'test samples')
            score = model.evaluate(features, Categorical_labels, show_accuracy=True, verbose=0)
            print('Test score:', score[0])
            print('Test accuracy:', score[1])
            return score[1]
        else:
            return model.predict_classes(features)


def IRNN():
    print("RNN")

    # trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy'
    # testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy - Copy'

    trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TrainSet'
    testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TestSet'

    trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2012\TrainSet'
    testFolder = 'C:\Users\l.valavanis\Desktop\Clef2012\TestSet'

    trainFolder = '/home/leonidas/Desktop/images/TrainSet'
    testFolder = '/home/leonidas/Desktop/images/TestSet'

    # trainFolder = '/home/leonidas/Desktop/images/SampleImages - Copy - Copy'
    # testFolder = '/home/leonidas/Desktop/images/SampleImages - Copy'

    [trainArray, train_labels, testArray, test_labels, nb_classes] = \
        load.load_dataset(trainFolder, testFolder, imageSize=(20, 20))

    # model = RNN().trainRNN(features=trainArray,labels=train_labels)
    model = RNN().trainRnnLSTM(features=trainArray, labels=train_labels)

    accuracy = RNN().predict(features=testArray, labels=test_labels, model=model)


# ---------------------------------------------------


def cnnTransfer():
    '''Transfer learning toy example:
    1- Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
    2- Freeze convolutional layers and fine-tune dense layers
       for the classification of digits [5..9].
    Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_transfer_cnn.py
    Get to 99.8% test accuracy after 5 epochs
    for the first five digits classifier
    and 99.2% for the last five digits after transfer + fine-tuning.
    '''

    import datetime

    np.random.seed(1337)  # for reproducibility

    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.utils import np_utils

    now = datetime.datetime.now

    batch_size = 128
    nb_classesFirst = 15
    nb_classesSecond = 15
    nb_epoch = 30

    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3




    def train_model(model, train, test, nb_classes):
        X_train = train[0].reshape(train[0].shape[0], train[0].shape[1], train[0].shape[2], train[0].shape[3])
        X_test = test[0].reshape(test[0].shape[0], test[0].shape[1], test[0].shape[2], test[0].shape[3])
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        Y_train = train[1]


        model.compile(loss='categorical_crossentropy', optimizer='adadelta')

        t = now()

        # create Image Generator
        datagen = createImageGenerator()
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)
        datagen.fit(X_test)

        [X_train, val_features, Y_train, val_labels] = matrix.Matrix().getValidationSet(trainArray=X_train,
                                                                                                train_labels=Y_train,
                                                                                                validationPercentage=.2)
        # reshape array for categorical cross entropy
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(Y_train, nb_classes)
        Y_test = np_utils.to_categorical(test[1], nb_classes)
        ValCategorical_labels = convertLabelsToCategorical(val_labels, nb_classes)

        early_stopping = EarlyStopping(monitor='val_loss', patience=6)


        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
                    validation_data=(val_features, ValCategorical_labels),
                    nb_worker=4, callbacks=[early_stopping])

        for e in range(nb_epoch):
            print ('Epoch %d/%d  ' % (e + 1, nb_epoch))
            batches = 0
            progbar = generic_utils.Progbar(X_train.shape[0])

            for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=batch_size):
                loss = model.train_on_batch(X_batch, Y_batch, accuracy=True)

                progbar.add(X_batch.shape[0])
                print(" - train loss: %.4f - train acc: %.4f " % (loss[0], loss[1]))
                batches += 1
                if batches > len(X_train) / batch_size:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break



        print('Training time: %s' % (now() - t))
        score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])


    trainFolder = '/home/leonidas/Documents/ImageClef2016/Clef2013/TrainSet'
    testFolder = '/home/leonidas/Documents/ImageClef2016/Clef2013/TestSet'

    trainFolder = '/home/leovala/databases/Clef2013/TrainSet'
    testFolder = '/home/leovala/databases/Clef2013/TestSet'


    [X_train, y_train, X_test, y_test, outputClasses] = \
        load.load_dataset(trainFolder, testFolder, imageSize=(100, 100))
    print(X_train.shape)
    print(X_test.shape)

    # # the data, shuffled and split between train and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # create two datasets one with digits below 5 and one with 5 and above
    X_train_lt5 = X_train[y_train < nb_classesFirst]
    y_train_lt5 = y_train[y_train < nb_classesFirst]
    X_test_lt5 = X_test[y_test < nb_classesFirst]
    y_test_lt5 = y_test[y_test < nb_classesFirst]

    X_train_gte5 = X_train[y_train >= nb_classesSecond]
    y_train_gte5 = y_train[y_train >= nb_classesSecond] - nb_classesFirst  # make classes start at 0 for
    X_test_gte5 = X_test[y_test >= nb_classesSecond]  # np_utils.to_categorical
    y_test_gte5 = y_test[y_test >= nb_classesSecond] - nb_classesFirst

    # define two groups of layers: feature (convolutions) and classification (dense)
    feature_layers = [
        Convolution2D(nb_filters, nb_conv, nb_conv,
                      border_mode='same',
                      input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])),
        Activation('relu'),
        Convolution2D(nb_filters, nb_conv, nb_conv),
        Activation('relu'),
        MaxPooling2D(pool_size=(nb_pool, nb_pool)),
        Dropout(0.3),
        Convolution2D(nb_filters * 2, nb_conv, nb_conv, border_mode='same'),
        Activation('relu'),
        Convolution2D(nb_filters * 2, nb_conv, nb_conv),
        Activation('relu'),
        MaxPooling2D(pool_size=(nb_pool, nb_pool)),
        Dropout(0.3),
        Flatten(),
    ]
    classification_layers = [

        Dense(512),
        Activation('relu'),
        Dropout(0.5),
        Dense(nb_classesFirst),
        Activation('softmax')

    ]

    # create complete model
    model = Sequential()
    for l in feature_layers + classification_layers:
        model.add(l)

    # train model for 5-digit classification [0..4]
    train_model(model,
                (X_train_lt5, y_train_lt5),
                (X_test_lt5, y_test_lt5), nb_classesFirst)

    # freeze feature layers and rebuild model
    for l in feature_layers:
        l.trainable = False

    # transfer: train dense layers for new classification task [5..9]
    train_model(model,
                (X_train_gte5, y_train_gte5),
                (X_test_gte5, y_test_gte5), nb_classesSecond)



# ---------------------------------------------------


def main():
    cnn()
    # mlp()
    # IRNN()

    # only method that has implementation in def
    # cnnTransfer()



if __name__ == '__main__':
    main()


