import sys

sys.path.append('../')
from Utils.Evaluation import measures
import LogisticRegression, SVM, MajorityClassifier
import matplotlib.pyplot as plt
import numpy as np
from MachineLearning.DeepLearning.Keras import Keras_test
import os


def plot_learning_curve(features_train, labels_train, features_test, labels_test, outputClasses=None, K="linear", C=1,
                        G=0.01,
                        method="error", classifier="SVM", a=0, b=1, showFigure=True, saveFigure=False, savePath=None,
                        nb_epochs=10):
    # run for every 10% of training set and compute training error and testing error
    step = len(features_train) / 10

    train = []
    test = []
    maj_clas = []

    for i in range(0, 10):
        print 'iteration : ', i

        # train for (i+1)*10 percent of training set
        f = features_train[0:((i + 1) * (step))]
        l = labels_train[0:((i + 1) * (step))]
        assert f.shape[0] == l.shape[0], 'Wrong number of input data! '

        if classifier == "SVM":
            # train classifier for the specific subset of training set
            model = SVM.train(f, l, k=K, c=C, g=G)

            # get training error
            predictionTrain = SVM.predict(f, model)

            # get testing error
            predictionTest = SVM.predict(features_test, model)

        elif classifier == "LR":
            # train classifier for the specific subset of training set
            model = LogisticRegression.train(f, l, c=C)

            # get training error
            predictionTrain = LogisticRegression.predict(f, model)

            # get testing error
            predictionTest = LogisticRegression.predict(features_test, model)
        elif classifier == "CNN":

            model = Keras_test.CNN().train(features=f, labels=l,outputClasses=outputClasses,
                                           learning_curves_OR_Cross_Val=True,nb_epoch=nb_epochs)
            # get training error
            predictionTrain = Keras_test.CNN().predictClasses(features=f, model=model)
            # get testing error
            predictionTest = Keras_test.CNN().predictClasses(features=features_test, model=model)

        # TODO : CNN MINIBATCHES LEARNING CURVES , Implementation : read 10% of data and train cnn.train with all the data
        # elif classifier == "CNN_minibatches":
        elif classifier == "MLP":

            model = Keras_test.MLP().train(features=f, labels=l, outputClasses=outputClasses,
                                           learning_curves_OR_Cross_Val=True, nb_epoch=nb_epochs)
            # get training error
            predictionTrain = Keras_test.MLP().predict(features=f, model=model, ShowAccuracy=False)
            # get testing error
            predictionTest = Keras_test.MLP().predict(features=features_test, model=model, ShowAccuracy=False)
        elif classifier == "SimpleRNN":

            model = Keras_test.RNN().trainRNN(features=f, labels=l, outputClasses=outputClasses, learning_curves=True)
            # get training error
            predictionTrain = Keras_test.RNN().predict(features=f, model=model, ShowAccuracy=False)
            # get testing error
            predictionTest = Keras_test.RNN().predict(features=features_test, model=model, ShowAccuracy=False)
        elif classifier == "RNN_LSTM":

            model = Keras_test.RNN().trainRnnLSTM(features=f, labels=l, outputClasses=outputClasses,
                                                  learning_curves=True)
            # get training error
            predictionTrain = Keras_test.RNN().predict(features=f, model=model, ShowAccuracy=False)
            # get testing error
            predictionTest = Keras_test.RNN().predict(features=features_test, model=model, ShowAccuracy=False)

        # get error for majority classifier
        predictionMajority = MajorityClassifier.predictMaj(labels_test)

        if method == "error":
            train.append(measures.error(l, predictionTrain))
            test.append(measures.error(labels_test, predictionTest))
            maj_clas.append(measures.error(labels_test, predictionMajority))
        elif method == "avgF1":
            train.append(measures.avgF1(l, predictionTrain, a, b))
            test.append(measures.avgF1(labels_test, predictionTest, a, b))
            maj_clas.append(measures.avgF1(labels_test, predictionMajority, a, b))

    print test[9]
    x = np.arange(len(train)) * 10

    plt.plot(x, train, color="blue", linewidth="2.0", label=classifier)
    plt.plot(x, test, color="blue", linestyle="dashed", linewidth="2.0")
    plt.plot(x, maj_clas, color="red", linewidth="2.0")
    plt.ylim(0, 1)
    plt.ylabel(method)
    plt.xlabel("% of messages")

    if method == "error":
        plt.legend(loc="upper left")
    elif method == "avgF1":
        plt.legend(loc="lower left")

    if saveFigure:
        assert savePath != None, "Give image path to save image"
        # with figure i can save it anywhere i want
        # fig1 = plt.gcf()
        plt.savefig(savePath)
        # clear current canvas . if we have show and save together we will have a problem...
        plt.clf()

    if showFigure:
        plt.show()


def plot_recall_precision(length, features_train, labels_train, features_test, labels_test):
    # threshold=[0.1 ,0.2 ,0.3 ,0.4,0.5,0.6,0.7,0.8,0.9]
    threshold = [x / 1000.0 for x in range(0, 1001, 1)]

    step = length / 3
    colors = ['b', 'r', 'g']
    for i in range(0, 3):

        # ((i+1)*(step)) percent of train data
        f = features_train[0:((i + 1) * (step))]
        l = labels_train[0:((i + 1) * (step))]

        # train classifier for the specific subset of training set
        model = LogisticRegression.train(f, l)

        # recall-precision for every threshold value
        recall = []
        precision = []

        for t in threshold:
            prediction = LogisticRegression.predict(features_test, model, t)

            recall.append(measures.recall(labels_test, prediction, 0))
            precision.append(measures.precision(labels_test, prediction, 0))

        plt.plot(recall, precision, linewidth="2.0", label=str((i + 1) * 33) + "% of train data", color=colors[i])

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Negative tweets')
    plt.legend()

    plt.show()


# find best cost in logistic regression
# cost comparison
def C_comparison(length, features_train, labels_train, features_test, labels_test):
    C = [0.001, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 10, 100, 350, 500, 1000, 3500, 5000, 10000, 50000, 100000]

    scores = []
    for c in C:
        model = LogisticRegression.train(features_train, labels_train, c)

        prediction = LogisticRegression.predict(features_test, model)

        scores.append((measures.avgF1(labels_test, prediction, 0, 1)))

    plt.plot(C, scores, color="blue", linewidth="2.0")
    plt.xticks(C)
    plt.ylabel("F1")
    plt.xlabel("C")
    plt.show()


def plotFeaturesF1(features_train, labels_train, features_test, labels_test):
    x = list(np.arange(len(features_train[0])))
    # x = list(np.arange(5))
    y = []
    for i in range(0, len(features_train[0])):
        f_train = features_train[:, i]
        f_test = features_test[:, i]
        f_train = f_train.reshape(f_train.shape[0], 1)
        f_test = f_test.reshape(f_test.shape[0], 1)
        model = LogisticRegression.train(f_train, labels_train)
        prediction = LogisticRegression.predict(f_test, model)
        y.append(measures.avgF1(labels_test, prediction, 0, 1))
    plt.plot(x, y, color="blue", linewidth="2.0")
    plt.ylabel("F1")
    plt.xlabel("# of Feature")
    plt.xticks(x)
    plt.show()
