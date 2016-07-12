import sys
sys.path.append('../../')
sys.path.append('../../MachineLearning')
sys.path.append('../Evaluation')
from MachineLearning import LogisticRegression
from Utils.Evaluation import measures

def findBestCost(features_train,labels_train,features_test,labels_test):
    C = [0.001,0.005,0.05,0.01,0.1,0.3,0.5,0.8,1,10,100,350,500,1000,3500,5000,10000,50000,100000]

    # import numpy
    # C = [x * 1 for x in numpy.arange(469, 471,0.1)]

    bestC =0
    bestAccuracy=0
    scores = []
    for c in C:
        model = LogisticRegression.train(features_train,labels_train,c)

        prediction = LogisticRegression.predict(features_test,model)

        scores.append((measures.avgF1(labels_test,prediction,0,1)))
        accuracy = measures.accuracy(labels_test,prediction)
        if accuracy >bestAccuracy:
            bestAccuracy=accuracy
            bestC=c


    return [bestC , bestAccuracy]

#return array C with elements starting from a to b with defined step
def getC(a,b,step):
    i=a
    C=[]
    while i>=a and i<=b:
        C.append(i)
        i+=step
    C = [float(format(x, '.5f')) for x in C]
    return C

