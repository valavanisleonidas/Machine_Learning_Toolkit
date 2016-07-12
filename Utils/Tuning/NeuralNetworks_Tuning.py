import sklearn.cross_validation as cross_val
import sys

sys.path.append('../../MachineLearning/DeepLearning/Keras')
from Keras_test import CNN, MLP


def neural_tuning(features_train, labels_train, features_test, labels_test,
                  outputClasses=None,
                  modelType="CNN",
                  batch_size=(10, 500, 50),
                  # CNN
                  nb_epoch=(10, 100, 20), nb_filters=(10, 50, 5), nb_pool=(2, 10, 1), nb_conv=(2, 10, 1),border_mode=['same' , 'valid'],
                  # MLP
                  activation=['relu', 'tanh'], num_input=(100, 1024, 100), depths=(1, 10, 1),
                  optimizers=['adam', 'sgd', 'rms', 'adadelta', 'adagrad'],
                  performCrossVal=False, K_Fold=10):
    bestAccuracy = -1
    bestBatchSize = []
    bestEpochs = []
    bestNumInputs = []
    bestDepth = []
    bestActivation = []
    bestOptimizer = []
    if modelType == "CNN":
        bestNbFilters = []
        bestNbPool = []
        bestNbConv = []
        bestBorderMode=[]
        for batch in range(batch_size[0], batch_size[1], batch_size[2]):
            for epochs in range(nb_epoch[0], nb_epoch[1], nb_epoch[2]):
                for filter in range(nb_filters[0], nb_filters[1], nb_filters[2]):
                    for poolSize in range(nb_pool[0], nb_pool[1], nb_pool[2]):
                        for conv in range(nb_conv[0], nb_conv[1], nb_conv[2]):
                            for activ in activation:
                                for opt in optimizers:
                                    for inputs in range(num_input[0], num_input[1], num_input[2]):
                                        for depth in range(depths[0], depths[1], depths[2]):
                                            for border in border_mode:
                                                print 'current batch :', str(batch)
                                                print 'current epochs :', str(epochs)
                                                print 'current filter :', str(filter)
                                                print 'current poolSize :', str(poolSize)
                                                print 'current conv :', str(conv)
                                                print 'current activ :', str(activ)
                                                print 'current optimizer :', str(opt)
                                                print 'current inputs :', str(inputs)
                                                print 'current depth :', str(depth)
                                                print 'current border :', str(border)
                                                if performCrossVal:
                                                    assert outputClasses != None, "Give Number of classes"

                                                    accuracy = neural_crossValidation(features_train=features_train,
                                                                                      labels_train=labels_train,
                                                                                      batch_size=batch,
                                                                                      nb_epoch=epochs,
                                                                                      nb_filters=filter,
                                                                                      nb_pool=poolSize,
                                                                                      nb_conv=conv,
                                                                                      outputClasses=outputClasses,
                                                                                      mode="CNN", K_Fold=K_Fold)
                                                else:
                                                    model = CNN().train(features=features_train, labels=labels_train,
                                                                        batch_size=batch,
                                                                        nb_epoch=epochs,
                                                                        nb_filters=filter,
                                                                        nb_conv=conv,
                                                                        nb_pool=poolSize,
                                                                        activation=activ,
                                                                        num_inputs=inputs,
                                                                        depth=depth,
                                                                        optimizer=opt,
                                                                        outputClasses=outputClasses,
                                                                        learning_curves_OR_Cross_Val=True,
                                                                        data_augmentation=True)
                                                    accuracy = CNN().predict(features=features_test, labels=labels_test, model=model)

                                                print 'accuracy :', str(accuracy)
                                                if accuracy >= bestAccuracy:
                                                    bestAccuracy = accuracy
                                                    bestBatchSize = batch
                                                    bestEpochs = epochs
                                                    bestNbFilters = filter
                                                    bestNbPool = poolSize
                                                    bestNbConv = conv
                                                    bestBorderMode=border
                                                    bestActivation = activ
                                                    bestNumInputs = inputs
                                                    bestDepth = depth
                                                    bestOptimizer = opt

                                                print '\nBest parameters so far:'
                                                print 'best acc:', bestAccuracy
                                                print 'best batch size:', bestBatchSize
                                                print 'best epochs:', bestEpochs
                                                print 'best filter:', bestNbFilters
                                                print 'best pool size:', bestNbPool
                                                print 'best conv:', bestNbConv
                                                print 'best activ :', bestActivation
                                                print 'best optimizer :', bestOptimizer
                                                print 'best inputs :', bestNumInputs
                                                print 'best depth :', bestDepth
                                                print 'best border :', bestBorderMode

        return [bestAccuracy, bestBatchSize, bestEpochs, bestNbFilters, bestNbPool, bestNbConv,bestBorderMode,bestActivation,bestNumInputs,bestDepth,bestOptimizer]

    if modelType == "MLP":

        for batch in range(batch_size[0], batch_size[1], batch_size[2]):
            for epochs in range(nb_epoch[0], nb_epoch[1], nb_epoch[2]):
                for activ in activation:
                    for opt in optimizers:
                        for inputs in range(num_input[0], num_input[1], num_input[2]):
                            for depth in range(depths[0], depths[1], depths[2]):
                                print 'current batch :', str(batch)
                                print 'current epochs :', str(epochs)
                                print 'current activ :', str(activ)
                                print 'current optimizer :', str(opt)
                                print 'current inputs :', str(inputs)
                                print 'current depth :', str(depth)

                                if performCrossVal:
                                    assert outputClasses != None, "Give Number of classes"
                                    accuracy = neural_crossValidation(features_train=features_train,
                                                                      labels_train=labels_train,
                                                                      batch_size=batch,
                                                                      nb_epoch=epochs,
                                                                      activation=activ,
                                                                      num_input=inputs,
                                                                      depth=depth,
                                                                      outputClasses=outputClasses,
                                                                      optimizer=opt,
                                                                      mode="MLP",
                                                                      K_Fold=K_Fold)
                                else:
                                    model = MLP().train(features=features_train, labels=labels_train,
                                                        nb_epoch=epochs,
                                                        activation=activ,
                                                        num_input=inputs,
                                                        depth=depth,
                                                        optimizer=opt,
                                                        batch_size=batch,
                                                        outputClasses=outputClasses,
                                                        learning_curves_OR_Cross_Val=True)
                                    accuracy = MLP().predict(features=features_test, labels=labels_test, model=model)

                                print 'accuracy :', str(accuracy)
                                if accuracy >= bestAccuracy:
                                    bestAccuracy = accuracy
                                    bestBatchSize = batch
                                    bestEpochs = epochs
                                    bestActivation = activ
                                    bestNumInputs = inputs
                                    bestDepth = depth
                                    bestOptimizer = opt

                                print 'Best parameters so far:'
                                print 'best acc:', bestAccuracy
                                print 'best batch size:', bestBatchSize
                                print 'best epochs:', bestEpochs
                                print 'best activation:', bestActivation
                                print 'best optimizer :', bestOptimizer
                                print 'bestNumInputs:', bestNumInputs
                                print 'bestDepth:', bestDepth

        return [bestAccuracy, bestBatchSize, bestEpochs, bestActivation, bestOptimizer, bestNumInputs, bestDepth]


def neural_crossValidation(features_train, labels_train, mode,
                           batch_size=100, nb_filters=32, nb_epoch=100, nb_pool=2, nb_conv=2,
                           optimizer='sgd', activation='relu', num_input=1024, depth=1,border_mode='same',
                           outputClasses=None, K_Fold=10):

    folds = cross_val.KFold(n=len(labels_train), n_folds=K_Fold, shuffle=True)
    print len(folds)
    bestAccuracy = []
    i = 0
    for train_index, test_index in folds:
        i +=1
        print 'Cross Validation iteration : ' , i
        if mode == "CNN":
            model = CNN().train(features=features_train[train_index], labels=labels_train[train_index],
                                nb_epoch=nb_epoch,
                                batch_size=batch_size,
                                nb_filters=nb_filters,
                                nb_pool=nb_pool,
                                nb_conv=nb_conv,
                                num_inputs=num_input,
                                depth=depth,
                                border_mode=border_mode,
                                activation=activation,
                                optimizer=optimizer,
                                data_augmentation=True,
                                outputClasses=outputClasses,
                                learning_curves_OR_Cross_Val=True)

            accuracy = CNN().predict(features=features_train[test_index], labels=labels_train[test_index], model=model,
                                     outputClasses=outputClasses,learning_curves_OR_Cross_Val=True)
        elif mode == "MLP":

            model = MLP().train(features=features_train[train_index], labels=labels_train[train_index],
                                outputClasses=outputClasses,
                                nb_epoch=nb_epoch,
                                batch_size=batch_size,
                                activation=activation,
                                num_input=num_input,
                                optimizer=optimizer,
                                depth=depth,
                                learning_curves_OR_Cross_Val=True)

            accuracy = MLP().predict(features=features_train[test_index], labels=labels_train[test_index],
                                     model=model,outputClasses=outputClasses,
                                     learning_curves_OR_Cross_Val=True)

    bestAccuracy.append(accuracy)
    # return mean of accuracies
    return float(sum(bestAccuracy)) / len(bestAccuracy) if len(bestAccuracy) > 0 else float('nan')
