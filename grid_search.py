import gc
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sys import argv
from os import mkdir


def read_file(number_of_fold, decomposition_level): 
    dataset = argv[1].upper()
    file = f"{dataset}/hyperparameter_optimization/lvl_{decomposition_level}/fold_{number_of_fold}_"
    file_train = file + 'train.sav'
    file_test = file + 'test.sav'
    train = joblib.load(file_train)
    test = joblib.load(file_test)
    return train['data'], test['data'], train['labels'], test['labels']


def calculateAccuracy(matrix):
    total = 0
    failures = 0
    for i in range(len(matrix)):
        total += sum(matrix[i])
        for j in range(len(matrix[0])):
            if i != j:
                failures += matrix[i,j]
    accuracy = ((total-failures)/total)*100
    return accuracy


def checkParameters(parametersToSave, savedParameters, ab=False):
    if ab:
        for i in savedParameters:
            if i[1:] == parametersToSave[1:] and type(i[0]) == type(parametersToSave[0]):
                return True
        return False
    else:
        if parametersToSave in savedParameters:
            return True
        return False
            

if __name__ == '__main__':
    if len(argv) != 2:
        print("The correct use is:\npython grid_search.py dataset_name\nDataset names:\n*deap\n*biomex-db")
    else:
        dataset = argv[1]
        if (dataset != "deap" and dataset != "biomex-db"):
            print("Dataset names:\n*deap\n*biomex-db")
        else:
            dataset = dataset.upper()
            mkdir(f"{dataset}/hyperparameter_optimization/results")
            for decomposition_level in range(2,6):
                mkdir(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}")
                
            #--------------------SVM------------------
            C = [0.5, 1, 10, 50, 100, 200, 300] 
            kernel = ['linear', 'rbf', 'sigmoid']
            tol = [1e-3, 0.1, 1, 1e-5, 1e-6, 1e-7] 
            gamma = ['scale', 'auto'] 
        
            for decomposition_level in range(2,6):
                try:
                    acc = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_svm.sav")
                    parameters = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_svm.sav")
                except:
                    acc = [] 
                    parameters = []
                for i in range(len(C)):
                    for j in range(len(kernel)):
                        for k in range(len(tol)):
                            for l in range(len(gamma)):
                                if checkParameters([C[i],kernel[j],tol[k],gamma[l]], parameters):
                                    continue
                                accuracies = []
                                for fold in range(10):
                                    X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                                    clf = SVC(C=C[i], kernel=kernel[j], tol=tol[k], gamma=gamma[l], random_state=2)
                                    clf.fit(X_train, y_train)
                                    y_pred = clf.predict(X_test) 
                                    accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                                acc.append(np.mean(accuracies))
                                parameters.append([C[i],kernel[j],tol[k],gamma[l]])
                                print(C[i],kernel[j],tol[k],gamma[l])
                                del accuracies, X_train, X_test, y_train, y_test, clf, y_pred
                                gc.collect()
                                joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_svm.sav")
                                joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_svm.sav")
                print(f"level-{decomposition_level} - DONE")
        
            
            #--------------------------RF---------------------    
            n_estimators = [1,10,50,100,200,500,750,1000] 
            criterion = ['gini', 'entropy'] 
            min_samples_split = [2,5,10,50,100] 
        
            for decomposition_level in range(2,6):
                try:
                    acc = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_rf.sav")
                    parameters = joblib.load(f"{dataset}/hyperparameter_optimization/resultss/lvl_{decomposition_level}/parameters_rf.sav")
                except:
                    acc = [] 
                    parameters = []
                for i in range(len(n_estimators)):
                    for j in range(len(criterion)):
                        for k in range(len(min_samples_split)):
                            if checkParameters([n_estimators[i],criterion[j],min_samples_split[k]], parameters):
                                continue
                            accuracies = []
                            for fold in range(10):
                                X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                                clf = RandomForestClassifier(n_estimators=n_estimators[i], criterion=criterion[j], min_samples_split=min_samples_split[k],random_state=2)
                                clf.fit(X_train,y_train)
                                y_pred = clf.predict(X_test) 
                                accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                            acc.append(np.mean(accuracies))
                            parameters.append([n_estimators[i],criterion[j],min_samples_split[k]])
                            print(n_estimators[i],criterion[j],min_samples_split[k])
                            del accuracies, X_train, X_test, y_train, y_test, clf, y_pred
                            gc.collect()
                            joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_rf.sav")
                            joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_rf.sav")
                print(f"level-{decomposition_level} - DONE")
                
            
            #-------------------KNN----------------------------
            n_neighbors = [1, 5, 10, 20, 50, 100] 
            leaf_size = [5, 10, 30, 50, 100] 
            p = [1,2] 
            
            for decomposition_level in range(2,6):
                try:
                    acc = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_knn.sav")
                    parameters = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_knn.sav")
                except:
                    acc = [] 
                    parameters = []
                for i in range(len(n_neighbors)):
                    for j in range(len(leaf_size)):
                        for k in range(len(p)):
                            if checkParameters([n_neighbors[i],leaf_size[j],p[k]], parameters):
                                continue
                            accuracies = []
                            for fold in range(10):
                                X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                                clf = KNeighborsClassifier(n_neighbors=n_neighbors[i],leaf_size=leaf_size[j],p=p[k])
                                clf.fit(X_train,y_train)
                                y_pred = clf.predict(X_test) 
                                accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                            acc.append(np.mean(accuracies))
                            parameters.append([n_neighbors[i],leaf_size[j],p[k]])
                            print(n_neighbors[i],leaf_size[j],p[k])
                            del accuracies, X_train, X_test, y_train, y_test, clf, y_pred
                            gc.collect()
                            joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_knn.sav")
                            joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_knn.sav")
                print(f"level-{decomposition_level} - DONE")
        
            
            #-------------------AB----------------------------
            svm = [SVC(C=100, kernel="rbf", tol=0.1, gamma="auto", random_state=2, probability=True),
                    SVC(C=100, kernel="rbf", tol=0.001, gamma="auto", random_state=2, probability=True),
                    SVC(C=50, kernel="rbf", tol=0.001, gamma="auto", random_state=2, probability=True),
                    SVC(C=50, kernel="rbf", tol=0.1, gamma="auto", random_state=2, probability=True)]
            rf = [RandomForestClassifier(n_estimators=750, criterion="entropy", min_samples_split=2, random_state=2),
                  RandomForestClassifier(n_estimators=750, criterion="entropy", min_samples_split=2, random_state=2),
                  RandomForestClassifier(n_estimators=1000, criterion="gini", min_samples_split=2, random_state=2),
                  RandomForestClassifier(n_estimators=500, criterion="entropy", min_samples_split=2, random_state=2)]
            base_estimator = [svm,rf] 
            n_estimators = [5,10,50,100,500,800] 
            learning_rate = [0.1,0.5,1,5] 
            algorithm = ['SAMME', 'SAMME.R']
            
            for decomposition_level in range(2,6):
                try:
                    acc = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_ab.sav")
                    parameters = joblib.load( f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_ab.sav")
                except:
                    acc = []
                    parameters = []
                for i in range(len(base_estimator)):
                    for j in range(len(n_estimators)):
                        for k in range(len(learning_rate)):
                            for l in range(len(algorithm)):
                                if checkParameters([base_estimator[i][decomposition_level-2],n_estimators[j],learning_rate[k],algorithm[l]], parameters, True):
                                    continue
                                else:
                                    accuracies = []
                                    for fold in range(10):
                                        X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                                        clf = AdaBoostClassifier(base_estimator=base_estimator[i][decomposition_level-2], n_estimators=n_estimators[j], learning_rate=learning_rate[k], algorithm=algorithm[l], random_state=2)
                                        clf.fit(X_train,y_train)
                                        y_pred = clf.predict(X_test) 
                                        accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                                    acc.append(np.mean(accuracies))
                                    parameters.append([base_estimator[i][decomposition_level-2],n_estimators[j],learning_rate[k],algorithm[l]])
                                    print([type(base_estimator[i][decomposition_level-2]),n_estimators[j],learning_rate[k],algorithm[l]])
                                    del accuracies, X_train, X_test, y_train, y_test, clf
                                    gc.collect()
                                    joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_ab.sav")
                                    joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_ab.sav")  
                print(f"level-{decomposition_level} - DONE")
        
            
            #--------------------------GNB---------------------    
            var_smoothing = [1e-9, 1-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10] 
            
            for decomposition_level in range(2,6):
                try:
                    acc = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_gnb.sav")
                    parameters = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_gnb.sav")
                except:
                    acc = [] 
                    parameters = []
                for i in range(len(var_smoothing)):
                    if checkParameters(var_smoothing[i], parameters):
                        continue
                    accuracies = []
                    for fold in range(10):
                        X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                        clf = GaussianNB(var_smoothing=var_smoothing[i])
                        clf.fit(X_train,y_train)
                        y_pred = clf.predict(X_test) 
                        accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                    acc.append(np.mean(accuracies))
                    parameters.append(var_smoothing[i])
                    print(var_smoothing[i])
                    del accuracies, X_train, X_test, y_train, y_test, clf, y_pred
                    gc.collect()
                    joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_gnb.sav")
                    joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_gnb.sav")
                print(f"level-{decomposition_level} - DONE")
