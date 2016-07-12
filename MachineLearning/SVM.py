#SVM Classifier
from sklearn import svm

#train model


def train(Train_Features,labels,g="auto",c=1,k="linear",coef0=0,degree=2):
    #define classifier
    if k=="linear":
        model = svm.LinearSVC(C=c,class_weight="balanced")
    elif k=="poly":
        model=svm.SVC(C=c,kernel=k,degree=degree,coef0=coef0)
    elif k=="rbf":
        model=svm.SVC(C=c,kernel=k,gamma=g,class_weight="balanced",cache_size=1000)

    #fit data
    model.fit(Train_Features,labels)

    return model

#predicts labels
def predict(Test_Features,model):
    return model.predict(Test_Features)

def predictProba(Test_Features,model):
    return model.decision_function(Test_Features)



def ManyClassifiers():

    import sys
    sys.path.append('../Utils')
    sys.path.append('../Utils/Evaluation')	
    import Load as load
    from Evaluation import measures
    from Utils.Tuning import SVM_Tuning

    # trainFolder = '/home/leonidas/Desktop/images/TrainSet'
    # testFolder = '/home/leonidas/Desktop/images/TestSet'
    #
    # trainFolder = '/home/leonidas/Desktop/images/SampleImages - Copy - Copy'
    # testFolder = '/home/leonidas/Desktop/images/SampleImages - Copy'
    #
    #
    # [trainArray, train_labels, testArray, test_labels,outputClasses] = matrix.load_dataset(trainFolder=trainFolder,testFolder=testFolder
    #                                                                                        ,imageSize=(250,250))
    # print (trainArray.shape)
    # print (testArray.shape)

    trainPath = '/home/leonidas/Downloads/Clef2013/train_2x2_CIELab_256.txt'
    testPath = '/home/leonidas/Downloads/Clef2013/test_2x2_CIELab_256.txt'
    trainLabelPath = '/home/leonidas/Downloads/Clef2013/train_2x2_CIELab_256_labels.txt'
    testLabelPath = '/home/leonidas/Downloads/Clef2013/test_2x2_CIELab_256_labels.txt'

    [trainArray, train_labels, testArray, test_labels,outputClasses] = load.loadFeatures(trainPath=trainPath, trainLabels=trainLabelPath, testPath=testPath,
                                   testLabels=testLabelPath);


    import numpy
    # resize to 2D array
    trainArray=numpy.reshape(trainArray,(trainArray.shape[0], -1))
    testArray=numpy.reshape(testArray,(testArray.shape[0], -1))
    print (trainArray.shape)
    print (testArray.shape)


    print outputClasses
    print ("Training")
    # train classifiers
    classifiers = []
    for i in range(0,outputClasses):
        temp_labels = [1 if x == i else 0 for x in train_labels]


        # print ('mpika tuning for classifier : {0}'.format(i))
        # [bestC , bestGamma]= SVM_Tuning.svm_tuning(trainArray,temp_labels,testArray,test_labels,randomized=False,kernel='linear')

        model = train(trainArray,temp_labels,g=16,c=16,k='linear')
        classifiers.append(model)

    print ("Testing")

    classifiers_predictions = []
    for model in classifiers:
        predictions = predictProba(testArray,model)
        classifiers_predictions.append(predictions)


    # classifiers_predictions = numpy.asarray(classifiers_predictions)
    # take maximum class probability from classifiers
    # predicted_labels = numpy.argmax(classifiers_predictions,axis=0)
    # print predicted_labels
    accuracy = measures.accuracy(test_labels,numpy.argmax(classifiers_predictions,axis=0) )
    print 'Accuracy with many classifiers : {0} '.format(accuracy)



def main():
    import sys
    sys.path.append('../Utils')
    sys.path.append('../Utils/Evaluation')
    sys.path.append('../Utils/Tuning')
    import Matrix as matrix
    import Load as load
    import numpy,learningCurves
    import measures
    import SVM_Tuning

    trainPath = '/home/leonidas/Downloads/Clef2013/train_2x2_CIELab_256.txt'
    testPath = '/home/leonidas/Downloads/Clef2013/test_2x2_CIELab_256.txt'
    trainLabelPath = '/home/leonidas/Downloads/Clef2013/train_2x2_CIELab_256_labels.txt'
    testLabelPath = '/home/leonidas/Downloads/Clef2013/test_2x2_CIELab_256_labels.txt'

    trainPath = '/home/leovala/databases/Clef2013/GBoCFeatures/train_2x2_CIELab_256.txt'
    testPath = '/home/leovala/databases/Clef2013/GBoCFeatures/test_2x2_CIELab_256.txt'
    trainLabelPath = '/home/leovala/databases/Clef2013/GBoCFeatures/train_2x2_CIELab_256_labels.txt'
    testLabelPath = '/home/leovala/databases/Clef2013/GBoCFeatures/test_2x2_CIELab_256_labels.txt'



    [trainArray, train_labels, testArray, test_labels,ouClasses] = load.loadFeatures(trainPath=trainPath, trainLabels=trainLabelPath, testPath=testPath,
                                   testLabels=testLabelPath);



    # resize to 2D array
    trainArray=numpy.reshape(trainArray,(trainArray.shape[0], -1))
    testArray=numpy.reshape(testArray,(testArray.shape[0], -1 ))
    print "training"


    # Random SEARCH ( 0 , 50 ):
    # Best parameters set found on development set:
    # {'C': 21, 'gamma': 46}
    #
    print 'tuning now...'
    [bestC , bestGamma]= SVM_Tuning.svm_tuning(trainArray,train_labels,testArray,test_labels,randomized=False)
    print bestC
    print bestGamma

    model = train(trainArray,train_labels,c=bestC,g=bestGamma,k='rbf')

    # model = train(trainArray,train_labels)

    print "testing"
    predictions = predict(testArray,model)
    accuracy = measures.accuracy(test_labels,predictions)
    print accuracy

    # Learning Curves
    learningCurves.plot_learning_curve(features_train=trainArray, labels_train=train_labels, features_test=testArray, labels_test=test_labels
                                       ,K='rbf',C=bestC,G=bestGamma)

    # one classifier for each category
    # ManyClassifiers()

if __name__ == '__main__':
    main()
