#Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression

#train model
def train(features,labels,c=1e5):
    #define classifier
    model = LogisticRegression(C=c)

    #fit the data
    model.fit(features, labels)

    return model

#predict labels
#default threshold=0.5
def predict(features,model,threshold=0.5):
    return model.predict(features)

    # predict for two classes
    # labels=[1 if posprob>threshold else 0 for negprob,posprob in model.predict_proba(features) ]
    # return labels


def main():


    import sys
    sys.path.append('../Utils')
    sys.path.append('../Utils/Evaluation')		
    import numpy
    from Evaluation import measures
    import Load as load
    from Tuning import Log_Regression_Tuning
    import learningCurves



    trainPath = '/home/leonidas/Downloads/Clef2013/train_2x2_CIELab_256.txt'
    testPath = '/home/leonidas/Downloads/Clef2013/test_2x2_CIELab_256.txt'
    trainLabelPath = '/home/leonidas/Downloads/Clef2013/train_2x2_CIELab_256_labels.txt'
    testLabelPath = '/home/leonidas/Downloads/Clef2013/test_2x2_CIELab_256_labels.txt'

    [trainArray, train_labels, testArray, test_labels,outputClasses] = load.loadFeatures(trainPath=trainPath, trainLabels=trainLabelPath, testPath=testPath,
                                   testLabels=testLabelPath);

    # resize to 2D array
    trainArray=numpy.reshape(trainArray,(trainArray.shape[0], -1 ))
    testArray=numpy.reshape(testArray,(testArray.shape[0],  -1 ))

    [bestC , bestAccuracy] = Log_Regression_Tuning.findBestCost(features_train=trainArray, labels_train=train_labels, features_test=testArray, labels_test=test_labels)
    print 'best accuracy : {0} , best cost : {1}'.format(bestAccuracy,bestC)
    model = train(trainArray,train_labels,c=bestC)


    predictions = predict(testArray,model)
    print predictions
    accuracy = measures.accuracy(test_labels,predictions)
    print ' acc : {0}'.format(accuracy)

    learningCurves.plot_learning_curve(features_train=trainArray, labels_train=train_labels, features_test=testArray, labels_test=test_labels,classifier="LR",C=bestC)

if __name__ == '__main__':
    main()
