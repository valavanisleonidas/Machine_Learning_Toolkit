from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils,generic_utils
import h5py
import numpy as np


def VGG_16(imageSize,imageChannels,outputClasses,weights_path=None):
    # 224,224,3 recommended SHAPE

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(imageChannels,imageSize[0],imageSize[1])))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2) ))
    print (model.output_shape)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print (model.output_shape)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print (model.output_shape)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    print (model.output_shape)

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    print (model.output_shape)

    assert (model.output_shape[2] != 0 and model.output_shape[3] != 0 ),\
            'Output size cannot be zero (shape must be at least (1,1) ). Change architecture to decrease convolution,' \
            ' pooling operations or increase size of shape'

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    if weights_path:
        print "loading weights"
        model.load_weights(weights_path)
        print "weights loaded !!"


    for layer in model.layers:
        layer.params = []
        layer.updates = []

    # ###################################### for testing
#    model.trainable_weights = True

    model.layers.pop()
    # model.params.pop()
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outputClasses, activation='softmax'))
    # ######################################


    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def createImageGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
                             samplewise_std_normalization=False,zca_whitening=False,rotation_range=0,
                             width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,vertical_flip=False):
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

def preprocess_image(image_path,img_width,img_height):
     from scipy.misc import imread, imresize, imsave
     im = imresize(imread(image_path), (img_width, img_height))
     im = im.transpose((2, 0, 1))
     im = np.expand_dims(im, axis=0)
     return im

def convertLabelsToCategorical(labels,outputClasses):
        labels = np.reshape(labels, (len(labels), 1))
        return  np_utils.to_categorical(labels, outputClasses)


def loadData():

    imageSize=(224,224)
    convertion = 'RGB'

    trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2012\TrainSet'
    testFolder = 'C:\Users\l.valavanis\Desktop\Clef2012\TestSet'

    trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy'
    testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy - Copy'

    trainFolder = '/home/leovala/databases/Clef2013//SampleImages - Copy - Copy'
    testFolder = '/home/leovala/databases/Clef2013/SampleImages - Copy'

    trainFolder = '/home/leonidas/Desktop/clef2016/Clef2013/SampleImages - Copy - Copy'
    testFolder = '/home/leonidas/Desktop/clef2016/Clef2013/SampleImages - Copy'


    trainFolder = '/home/leonidas/Desktop/clef2016/SampleImages - Copy - Copy'
    testFolder = '/home/leonidas/Desktop/clef2016/SampleImages - Copy'

#    trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy'
#    testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy - Copy'
#
#    trainFolder = '/home/leovala/databases/Clef2013/TrainSet'
#    testFolder = '/home/leovala/databases/Clef2013/TestSet'

    [trainArray, train_labels, testArray, test_labels,outputClasses] = \
    load.load_dataset(trainFolder, testFolder,imageSize=imageSize,convertion=convertion)


    [trainArray, val_features, Trainlabels, val_labels] = matrix.Matrix().SplitTrainValidationArrays(trainArray=trainArray,train_labels=train_labels,validationPercentage=.2)
    # reshape array for categorical cross entropy
    ValCat_labels = convertLabelsToCategorical(val_labels,outputClasses)
    CatTrain_labels = convertLabelsToCategorical(Trainlabels,outputClasses)
    CatTest_labels = convertLabelsToCategorical(test_labels,outputClasses)


    return [trainArray ,CatTrain_labels,testArray, CatTest_labels,val_features,ValCat_labels,outputClasses]

if __name__ == "__main__":
    import sys
    sys.path.append('../../../')

    import numpy as np
    from Utils import Load as load
    from Utils import Matrix as matrix
    nb_epoch = 10
    batch_size = 128

    [trainArray ,CatTrain_labels,testArray, CatTest_labels,val_features,ValCat_labels,outputClasses] = loadData()



    print "Creating Model..."
    # Create Model using weights
    model = VGG_16(imageChannels=trainArray.shape[1],
                   imageSize=(trainArray.shape[2],trainArray.shape[3]),
                   outputClasses=outputClasses,
                   weights_path='/home/leonidas/Desktop/clef2016/vgg16_weights.h5')
    print "Training..."
    ##################################################################
    # TRAIIIIIIIIIIIIN
    # create Image Generator
    datagen = createImageGenerator()
    datagen.fit(trainArray)
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    print "generator"
    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(trainArray, CatTrain_labels, batch_size=batch_size),
                        samples_per_epoch=trainArray.shape[0],
                        nb_epoch=nb_epoch, show_accuracy=True, verbose=1,
                        validation_data=(val_features, ValCat_labels),
                        nb_worker=4, callbacks=[early_stopping])


    for e in range(nb_epoch):
        print('Epoch %d/%d  ' % (e + 1, nb_epoch))
        batches = 0
        progbar = generic_utils.Progbar(trainArray.shape[0])

        for X_batch, Y_batch in datagen.flow(trainArray, CatTrain_labels, batch_size=batch_size):
            loss = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(X_batch.shape[0])
            print(" - train loss: %.4f - train acc: %.4f " % (loss[0], loss[1]))
            batches += 1
            if batches > len(trainArray) / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
    ##################################################################


    print "testing"
    ##################################################################
    # TEEEEEEEEEEEEEEEEEEEST
    score = model.evaluate(testArray, CatTest_labels, show_accuracy=True, verbose=1)
    print score