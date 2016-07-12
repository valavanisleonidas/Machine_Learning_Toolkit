import os
import platform
import numpy


class Matrix:
    def __init__(self):
        if platform.system() == "Windows":
            self.delimiterForPath = "\\"
        else:
            self.delimiterForPath = "/"

        self.labelsDType = numpy.int32
        self.imagesDType = numpy.float32

    def deleteRows(self, array, rows, axis):
        return numpy.delete(array, rows, axis)

    def swapAxes(self, array, axe1, axe2):
        return numpy.swapaxes(array, axe1, axe2)

    def getImageCategoryFromPath(self, imagePath):
        # path in format : ..\\Category\\ImageName

        return numpy.array(imagePath.split(self.delimiterForPath, len(imagePath))[
                               len(imagePath.split(self.delimiterForPath, len(imagePath))) - 2], dtype=self.labelsDType)

    def getNumberOfClasses(self, array):
        return len(numpy.unique(array))

    def getImagesInDirectory(self, folderPath, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        imagesList = []
        assert os.path.isdir(folderPath), 'No folder with that name exists : %r ' % folderPath
        # for all images in folder path
        for root, dirs, files in os.walk(folderPath):
            for name in files:
                if name.endswith(extensions):
                    imagesList.append(root + self.delimiterForPath + name)
        return imagesList

    def addDimension(self, array, axis):
        return numpy.expand_dims(a=array, axis=axis)

    def ExtractImages(self, folderPath, image_size=(256, 256), convertion=None, imageChannels=3,
                  preprocessImages=False ,normalize=True ,normalizeRange=(0,1) ):
        from Images.ImageProcessing import ImageProcessing

        imageList = self.getImagesInDirectory(folderPath=folderPath)
        assert len(imageList) > 0, 'No images in folder : %r' % folderPath
        if convertion != "Grayscale" and imageChannels != 3:
            if convertion == None:
                convertion = "RGB"
            raise ValueError(' %r supports only 3 image channels!' % convertion)

        images_list = []
        labels_list = []
        # for all images in folder path
        for imagePath in imageList:
            # get category of image and add category to array
            labels_list.append(
                self.getImageCategoryFromPath(imagePath=imagePath))
            # get image array and add image to array
            images_list.append(
                ImageProcessing().getImageArray(imagePath=imagePath, imageSize=image_size, convertion=convertion,
                                                imageChannels=imageChannels,preprocessImages=preprocessImages,
                                                Normalize=normalize,NormalizeRange=normalizeRange))

        # convert lists to numpy array
        allLabelsArray = numpy.array(labels_list).reshape(len(labels_list))
        allImagesArray = numpy.array(images_list).reshape(len(imageList), imageChannels, image_size[0], image_size[1])

        return [allImagesArray, allLabelsArray]

    # returns batches from data with size batchSize
    def chunker(self,data, batchSize):
        return (data[pos:pos + batchSize] for pos in xrange(0, len(data), batchSize))

    def shuffleMatrix(self,array):
        numpy.random.shuffle(array)

    def shuffleMatrixAlongWithLabels(self, array1, array2):
        # shuffle array1 (images) with corresponding labels array2
        from random import shuffle

        array1_shuf = []
        array2_shuf = []
        index_shuf = range(len(array1))
        shuffle(index_shuf)
        for i in index_shuf:
            array1_shuf.append(array1[i])
            array2_shuf.append(array2[i])
        return [numpy.array(array1_shuf, dtype=self.imagesDType).astype('float32'), numpy.array(array2_shuf, dtype=self.labelsDType).astype('float32')]

    def TakeExamplesFromEachCategory(self,features,labels,maxImagesPerCategory=10):
        import gc
        import os

        validationArray = []
        validation_labels=[]
        # for 0 to number of output classes
        for index in range(0,self.getNumberOfClasses(labels)):
            print ('mpika 1')
            # find indexes of category index
            indexes = numpy.where(labels == index)
            # if train has 1 instance don't take it for validation
            if len(indexes[0]) in [ 0 , 1 ]:
                continue
            # if instances are less than max categories given
            if len(indexes[0]) <= maxImagesPerCategory:
                # take half for validation
                maxImagesPerCategory= len(indexes[0])/2
            print ('mpika 2')
            assert len(indexes[0]) >= maxImagesPerCategory ,\
                "Error : Validation examples per category more than train instances. Category: {0}" \
                " validation pes category : {1} , training examples : {2} ".format(index,maxImagesPerCategory,len(indexes[0]),)

            count = 0
            # for indexes in category
            for catIndex in indexes[0]:
                print ('mpika 3')
                count +=1
                if count > maxImagesPerCategory:
                    print ('mpika 3.1')
                    break
                print ('mpika 3.2')
                validationArray.append(features[catIndex])
                print ('mpika 3.3')
                validation_labels.append(labels[catIndex ])
                print ('mpika 3.4 catIndex' , catIndex)
                features = numpy.delete(features,catIndex,axis=0)
                print ('mpika 3.5')
                labels = numpy.delete(labels,catIndex,axis=0)
                print ('mpika 3.6')
                gc.collect()
        print ('mpika 4')
        return [features, numpy.array(validationArray,dtype=self.imagesDType).astype('float32'), labels,
                numpy.array(validation_labels,dtype=self.labelsDType).astype('int32')]

    def takeLastExamples(self,trainArray, train_labels, validationPercentage=.2):
        # take validationPercentage of training data for validation
        validationExamples = int(validationPercentage * len(trainArray))

        # We reserve the last validationExamples training examples for validation.
        trainArray, validationArray = trainArray[:-validationExamples], trainArray[-validationExamples:]
        train_labels, validation_labels = train_labels[:-validationExamples], train_labels[-validationExamples:]

        return [trainArray, validationArray, train_labels, validation_labels]

    def SplitTrainValidation(self, trainArray, train_labels, validationPercentage=.2,takeLastExamples=False,maxImagesPerCategory=10):
        if takeLastExamples:
            return self.takeLastExamples(trainArray, train_labels, validationPercentage)
        else:
            return self.TakeExamplesFromEachCategory(trainArray, train_labels,maxImagesPerCategory)

    def moveFile(self, src, dest):
        import shutil
        shutil.move(src, dest)

if __name__ == '__main__':

    trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TrainSet'
    testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\TestSet'
    #
    # trainFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy'
    # testFolder = 'C:\Users\l.valavanis\Desktop\Clef2013\SampleImages - Copy - Copy'
    #
    # # trainFolder = '/home/leonidas/Desktop/images/train'
    # # testFolder = '/home/leonidas/Desktop/images/test'
    #
    # [trainArray, train_labels, testArray, test_labels, validationArray, validation_labels, outputClasses] = \
    # load_dataset(trainFolder, testFolder,imageSize=(3,3),convertion='L',imageChannels=1)
    #
    # print trainArray.shape
    # print trainArray
    # # print validation_labels
    # # print train_labels
    # # print trainArray
    #
    # print trainArray.shape
    # print train_labels.shape
    # print testArray.shape
    # print test_labels.shape
    # print validationArray.shape
    # print validation_labels.shape
    #
    # trainPath = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\Features\\train_2x2_CIELab_512.txt'
    # testPath = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\Features\\test_2x2_CIELab_512.txt'
    # trainLabelPath = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\Features\\train_2x2_CIELab_512_labels.txt'
    # testLabelPath = 'C:\\Users\\l.valavanis\\Desktop\\Clef2013\\GBoC\Features\\test_2x2_CIELab_512_labels.txt'

    # [trainArray, train_labels, testArray, test_labels, validationArray, validation_labels,
    # outputClasses] = loadFeatures(trainPath=trainPath, trainLabels=trainLabelPath, testPath=testPath,
    #                                testLabels=testLabelPath);
    i=0;
    for trainArray,train_labels in Matrix().getArrayOfImagesUsingMiniBatches(folderPath=trainFolder,image_size=(100,100),batch_size=15):
        print (trainArray.shape)
        print (train_labels.shape)
        i+=len(trainArray)

    print "aaasdasdas d  : ",i
    # # print validation_labels
    # # print train_labels
    # # print trainArray
    #
    # print trainArray.shape
    # print train_labels.shape
    # print testArray.shape
    # print test_labels.shape
    # print validationArray.shape
    # print validation_labels.shape

