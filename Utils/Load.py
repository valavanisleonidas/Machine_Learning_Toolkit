import numpy
import Matrix as matrix


def load_dataset(trainFolder, testFolder, imageSize=(256, 256), convertion=None, imageChannels=3,
                 splitToValidationSet=True,takeLastExamples=True, normalize=True, normalizeRange=(0, 1),maxImagesPerCategory=10):

    [trainArray, train_labels] = matrix.Matrix().ExtractImages(folderPath=trainFolder, image_size=imageSize,
                                              convertion=convertion, imageChannels=imageChannels,
                                              normalize=normalize,normalizeRange=normalizeRange)
    print "Finished train"

    [testArray, test_labels] = matrix.Matrix().ExtractImages(folderPath=testFolder, image_size=imageSize,
                                               convertion=convertion, imageChannels=imageChannels,
                                               normalize=normalize,normalizeRange=normalizeRange)
    print "Finished test"

    # shuffle train data and labels
    [trainArray, train_labels] = matrix.Matrix().shuffleMatrixAlongWithLabels(trainArray, train_labels)
    # get number of classes
    outputClasses = matrix.Matrix().getNumberOfClasses(train_labels)




    if splitToValidationSet:
        print ("mpika edw")
        [trainArray, validationArray, train_labels, validation_labels] = matrix.Matrix().SplitTrainValidation(
            trainArray=trainArray, train_labels=train_labels,takeLastExamples=takeLastExamples,maxImagesPerCategory=maxImagesPerCategory)
        print ("eftasa edw")
        return [trainArray, train_labels, testArray, test_labels, validationArray, validation_labels, outputClasses]
    else:
        return [trainArray, train_labels, testArray, test_labels, outputClasses]


def loadFeatures(trainPath, testPath, trainLabels, testLabels=None, splitToValidationSet=False):
    from Files import FilesIO
    [trainArray, testArray, train_labels, test_labels] = FilesIO.FilesIO().getFilesWithLabelsInDifferentFile \
        (trainPath, testPath, trainLabels, testLabels)

    trainArray = matrix.Matrix().addDimension(array=trainArray, axis=2)
    testArray = matrix.Matrix().addDimension(array=testArray, axis=2)
    trainArray = numpy.swapaxes(trainArray, 1, 2)
    testArray = numpy.swapaxes(testArray, 1, 2)

    # shuffle train data and labels
    [trainArray, train_labels] = matrix.Matrix().shuffleMatrixAlongWithLabels(trainArray, train_labels)
    # get number of classes
    outputClasses = matrix.Matrix().getNumberOfClasses(train_labels)

    if splitToValidationSet:
        [trainArray, validationArray, train_labels, validation_labels] = matrix.Matrix().SplitTrainValidation(
            trainArray=trainArray, train_labels=train_labels)
        return [trainArray, train_labels, testArray, test_labels, validationArray, validation_labels, outputClasses]
    else:
        return [trainArray, train_labels, testArray, test_labels, outputClasses]


def load_dataset_minibatches(folderPath, image_size=(256, 256), convertion=None, imageChannels=3, batch_size=32,
                             Shuffle=True, preprocessImages=False, normalize=True, normalizeRange=(0, 1)):
    from Images.ImageProcessing import ImageProcessing
    # Check input parameters
    imageList = matrix.Matrix().getImagesInDirectory(folderPath=folderPath)
    assert len(imageList) > 0, 'No images in folder : %r' % folderPath
    if convertion != "Grayscale" and imageChannels != 3:
        if convertion == None:
            convertion = "RGB"
        raise ValueError(' %r supports only 3 image channels!' % convertion)

    if Shuffle:
        # shuffle Images
        matrix.Matrix().shuffleMatrix(imageList)
        # numpy.random.shuffle(imageList)

    # for all images in imageList
    for ImagesBatch in matrix.Matrix().chunker(imageList, batch_size):
        images_list = []
        labels_list = []
        # for all images in batchSize
        for imagePath in ImagesBatch:
            # get category of image and add category to array
            labels_list.append(
                matrix.Matrix().getImageCategoryFromPath(imagePath=imagePath))
            # get image array and add image to array
            images_list.append(
                ImageProcessing().getImageArray(imagePath=imagePath, imageSize=image_size, convertion=convertion,
                                                imageChannels=imageChannels,preprocessImages=preprocessImages,
                                                Normalize=normalize,NormalizeRange=normalizeRange))
        # convert lists to numpy array
        BatchLabelsArray = numpy.array(labels_list).reshape(len(ImagesBatch))
        BatchImagesArray = numpy.array(images_list).reshape(len(ImagesBatch), imageChannels, image_size[0],
                                                            image_size[1])

        yield BatchImagesArray, BatchLabelsArray
