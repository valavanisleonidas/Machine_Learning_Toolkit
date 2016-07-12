import numpy
import os
class FilesIO:

    def __init__(self):
        pass

    def writeArrayToFile(self, filename, array, arrayType='%10.5f', delimiter=','):
        numpy.savetxt(filename, array, fmt=arrayType, delimiter=delimiter)

    def readTextFile(self, filePath, type=float):
        return numpy.asarray(numpy.loadtxt(filePath), dtype=type)

    def getFilesWithLabelsInDifferentFile(self, trainPath, testPath, labelTrainPath, labelTestPath):
        assert os.path.exists(trainPath) , 'Train file does not exist %r' % trainPath
        assert os.path.exists(testPath) , 'Test file does not exist %r' % testPath
        assert os.path.exists(labelTrainPath) , 'Labels Train file does not exist %r' % labelTrainPath
        assert os.path.exists(labelTestPath) , 'Labels Test file does not exist %r' % labelTestPath


        train = self.readTextFile(trainPath)
        test = self.readTextFile(testPath)
        labelTrain = self.readTextFile(labelTrainPath, int)
        labelTest = self.readTextFile(labelTestPath, int)

        return [train, test, labelTrain, labelTest]

    def getFilesWithLabelsInTheSameFile(self, trainPath, testPath):
        assert os.path.exists(trainPath) , 'Train file does not exist %r' % trainPath
        assert os.path.exists(testPath) , 'Test file does not exist %r' % testPath

        train = self.readTextFile(trainPath)
        test = self.readTextFile(testPath)

        # remove labels from txt (in first column)
        labelTrain = numpy.asarray(train[:, 0], dtype=int)
        labelTest = numpy.asarray(test[:, 0], dtype=int)
        # remove first column (labels ) from array
        train = numpy.asarray([line[1:] for line in train])
        test = numpy.asarray([line[1:] for line in test])

        return [train, test, labelTrain, labelTest]