import sys
sys.path.append('../Evaluation')	
sys.path.append('../../MachineLearning')
from Utils.Evaluation import measures
from MachineLearning import LogisticRegression, SVM
import numpy as np
from sklearn import svm,linear_model,cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


def custom_optimizer(features_train,labels_train,features_test,labels_test,C):
    # test classifier for different values

    #C = [x/float(1000) for x in range(1,1000)]
    scores = []

    for c in C:
        #print str(C.index(c)*100/float(len(C)))+"%"
        s=0

        #3 evals
        for x in range(0,3):
            model = SVM.train(features_train,labels_train,g=1,c=c,k="linear")
            prediction = SVM.predict(features_test,model)
            score = measures.avgF1(labels_test,prediction,0,1)
            s+=score
            
        s = s/float(3)
        scores.append(s)
        
        print "c = "+str(c)+" score = "+str(score)

    bestScore = max(scores)
    bestC = C[scores.index(bestScore)]
    print "best C = "+str(bestC)+" , avgF1 = "+str(bestScore)
    return [bestC , bestScore]

def KFoldCrossValidation(K_Folds,features_train,labels_train):
    scores = cross_validation.cross_val_score(svm.SVC(kernel="rbf",cache_size=1000,class_weight="balanced"), features_train, labels_train, cv=K_Folds, n_jobs=-1, scoring="accuracy")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def svm_tuning(features_train,labels_train,features_test,labels_test,kernel="rbf",C=[],gamma=[],randomized=False,i=50):
    if C==[]:
        C = [x * 1 for x in range(1, 50)];
    if gamma==[]:
        gamma = [x * 1 for x in range(1, 50)];
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(features_train, labels_train, test_size=0.25, random_state=0)

    if randomized:
        if kernel == "linear":
            tuned_parameters = {"C":C}
            clf =  RandomizedSearchCV(svm.LinearSVC(class_weight="balanced"), param_distributions=tuned_parameters, cv=5,scoring="accuracy",n_iter=i)
        elif kernel == "rbf":
            tuned_parameters = {'C': C, 'gamma': gamma}
            clf = RandomizedSearchCV(svm.SVC(kernel="rbf",cache_size=1000,class_weight="balanced"), param_distributions=tuned_parameters, cv=5,scoring="accuracy",n_iter=i)
        elif kernel == "logistic":
            tuned_parameters = {"C":C}
            clf =  RandomizedSearchCV(linear_model.LogisticRegression(), param_distributions=tuned_parameters, cv=5,scoring="accuracy",n_iter=i)
    else:
        if kernel == "linear":
            tuned_parameters = [{'C': C}]
            clf = GridSearchCV(svm.LinearSVC(class_weight="balanced"), tuned_parameters, cv=5,scoring="accuracy")
        elif kernel == "rbf":
            tuned_parameters = [{'C': C, 'gamma': gamma}]
            clf = GridSearchCV(svm.SVC(kernel="rbf",cache_size=1000,class_weight="balanced"), tuned_parameters, cv=5,scoring="accuracy")
        elif kernel == "logistic":
            tuned_parameters = [{'C': C}]
            clf = GridSearchCV(linear_model.LogisticRegression(), tuned_parameters, cv=5,scoring="accuracy")

    clf.fit(X_train, y_train)

    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print("Best parameters set found on development set:")
    print(clf.best_params_)

    y_true, y_pred = y_test, clf.predict(X_test)
    #print(classification_report(y_true, y_pred))
    print(measures.avgF1(np.array(y_true),y_pred,0,1))

    print("FINAL C:")
    bestC = clf.best_params_["C"]
    if kernel == "linear":
        model = SVM.train(features_train,labels_train,c=bestC,k="linear")
    elif kernel == "rbf":
        bestGamma = clf.best_params_["gamma"]
        model = SVM.train(features_train,labels_train,c=bestC,g=bestGamma,k="rbf")
    elif kernel == "logistic":
        model = LogisticRegression.train(features_train,labels_train,c=bestC)


    prediction = SVM.predict(features_test,model)
    print(measures.avgF1(labels_test,prediction,0,1))
    print(" ")
    if kernel == "rbf":
        return [bestC,bestGamma]
    else:
        return bestC




#calculates average f1 score
def custom_scorer(estimator, X, y):
	prediction = estimator.predict(X)
	return measures.avgF1(y,prediction,0,1)




def train_svm(data, labels,C):
    #model = svm.LinearSVC(C=C)
    #print C
    #model = svm.SVC(C=C,kernel="linear",cache_size=2000)
    #model = linear_model.SGDClassifier()
    #model.fit(data, labels)
    model = SVM.train(data,labels,c=C,k="linear")
    return model

def performance(x_train, y_train, x_test, y_test, n_neighbors=None, n_estimators=None, max_features=None,
                kernel=None, C=None, gamma=None, degree=None, coef0=None):

    #train model
    model = train_svm(x_train, y_train, C)
    # predict the test set
    #predictions = model.decision_function(x_test)
    predictions = SVM.predict(x_test,model)
   
    #return optunity.metrics.roc_auc(y_test, predictions, positive=True)
    return measures.avgF1(y_test,predictions,0,1)

