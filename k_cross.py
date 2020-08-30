import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sys import argv
from os import mkdir

def read_file(dataset, level_of_decomposition, time, number_of_fold): 
    file = f"{dataset.upper()}/experiment/lvl_{level_of_decomposition}/{time}/fold_{number_of_fold}_"
    file_train = f"{file}train.sav"
    file_test = f"{file}test.sav"
    train = joblib.load(file_train)
    test = joblib.load(file_test)
    return train['data'], test['data'], train['labels'], test['labels']


def classification(classifier_type, level_of_decomposition, dataset, X_train, y_train, X_test, y_test):
    if classifier_type == 1:
        if level_of_decomposition == 2:
            clf = SVC(C=100, kernel='rbf', probability=True, tol=0.1, gamma='auto', random_state=2) if dataset == "deap" else SVC(C=300, kernel='linear', probability=True, tol=0.001, gamma='scale', random_state=2)
        elif level_of_decomposition == 3:
            clf = SVC(C=100, kernel='rbf', probability=True, tol=0.001, gamma='auto', random_state=2) if dataset == "deap" else SVC(C=300, kernel='linear', probability=True, tol=0.001, gamma='scale', random_state=2)
        elif level_of_decomposition == 4:
            clf = SVC(C=50, kernel='rbf', probability=True, tol=0.001, gamma='auto', random_state=2) if dataset == "deap" else SVC(C=300, kernel='linear', probability=True, tol=0.1, gamma='scale', random_state=2)
        elif level_of_decomposition == 5:
            clf = SVC(C=50, kernel='rbf', probability=True, tol=0.1, gamma='auto', random_state=2) if dataset == "deap" else SVC(C=300, kernel='linear', probability=True, tol=0.1, gamma='scale', random_state=2)
    elif classifier_type == 2:
        if level_of_decomposition == 2:
            clf = RandomForestClassifier(n_estimators=750, criterion='entropy', min_samples_split=2, random_state=2) if dataset == "deap" else RandomForestClassifier(n_estimators=500, criterion='gini', min_samples_split=2, random_state=2)
        elif level_of_decomposition == 3:
            clf = RandomForestClassifier(n_estimators=750, criterion='entropy', min_samples_split=2, random_state=2) if dataset == "deap" else RandomForestClassifier(n_estimators=750, criterion='entropy', min_samples_split=2, random_state=2)
        elif level_of_decomposition == 4:
            clf = RandomForestClassifier(n_estimators=1000, criterion='gini', min_samples_split=2, random_state=2) if dataset == "deap" else RandomForestClassifier(n_estimators=750, criterion='gini', min_samples_split=2, random_state=2)
        elif level_of_decomposition == 5:
            clf = RandomForestClassifier(n_estimators=500, criterion='entropy', min_samples_split=2, random_state=2) if dataset == "deap" else RandomForestClassifier(n_estimators=750, criterion='entropy', min_samples_split=2, random_state=2)
    elif classifier_type == 3:
        if level_of_decomposition == 2:
            clf = KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=2, n_jobs=-1) if dataset == "deap" else KNeighborsClassifier(n_neighbors=5, leaf_size=5, p=1, n_jobs=-1)
        elif level_of_decomposition == 3:
            clf = KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=2, n_jobs=-1) if dataset == "deap" else KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=1, n_jobs=-1)
        elif level_of_decomposition == 4:
            clf = KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=2, n_jobs=-1) if dataset == "deap" else KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=1, n_jobs=-1)
        elif level_of_decomposition == 5:
            clf = KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=2, n_jobs=-1) if dataset == "deap" else KNeighborsClassifier(n_neighbors=1, leaf_size=5, p=1, n_jobs=-1)
    elif classifier_type == 4:
        if level_of_decomposition == 2:
            clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=750, criterion='entropy', min_samples_split=2, random_state=2), n_estimators=5, algorithm='SAMME', learning_rate=0.1, random_state=2) if dataset == "deap" else AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=500, criterion='gini', min_samples_split=2, random_state=2), n_estimators=5, algorithm='SAMME', learning_rate=0.1, random_state=2) 
        elif level_of_decomposition == 3:
            clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=750, criterion='entropy', min_samples_split=2, random_state=2), n_estimators=5, algorithm='SAMME', learning_rate=0.1, random_state=2) if dataset == "deap" else AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=750, criterion='entropy', min_samples_split=2, random_state=2), n_estimators=5, algorithm='SAMME', learning_rate=0.1, random_state=2)
        elif level_of_decomposition == 4:
            clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=1000, criterion='gini', min_samples_split=2, random_state=2), n_estimators=5, algorithm='SAMME', learning_rate=0.1, random_state=2) if dataset == "deap" else AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=750, criterion='gini', min_samples_split=2, random_state=2), n_estimators=5, algorithm='SAMME', learning_rate=0.1, random_state=2)
        elif level_of_decomposition == 5:
            clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=500, criterion='entropy', min_samples_split=2, random_state=2), n_estimators=5, algorithm='SAMME', learning_rate=0.1, random_state=2) if dataset == "deap" else AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=750, criterion='entropy', min_samples_split=2, random_state=2), n_estimators=5, algorithm='SAMME', learning_rate=0.1, random_state=2)
    elif classifier_type == 5:
        if level_of_decomposition == 2:
            clf = GaussianNB(var_smoothing=0.01) if dataset == "deap" else GaussianNB(var_smoothing=1e-9)
        elif level_of_decomposition == 3:
            clf = GaussianNB(var_smoothing=0.001) if dataset == "deap" else GaussianNB(var_smoothing=1e-7)
        elif level_of_decomposition == 4:
            clf = GaussianNB(var_smoothing=0.001) if dataset == "deap" else GaussianNB(var_smoothing=1e-9)
        elif level_of_decomposition == 5:
            clf = GaussianNB(var_smoothing=0.01) if dataset == "deap" else GaussianNB(var_smoothing=1e-9)
    
    clf.fit(X_train,y_train) 
    y_pred = clf.predict(X_test) 
    return confusion_matrix(y_test,y_pred)


def calculateAccuracy(matrix):
    accuracyBySubject = []
    total = 0
    for i in range(len(matrix)):
        total += sum(matrix[i])
    
    for s in range(len(matrix)):
        tp = matrix[s][s]
        fp = 0
        tn = 0
        for i in range(len(matrix)):
            if i != s:
                fp += matrix[i][s]
                tn +=sum(matrix[i])
        tn -= fp
        accuracyBySubject.append((tp+tn)/total)        
    accuracy = np.mean(accuracyBySubject)*100
    return accuracy


def calculateSensitivity(matrix):
    sensitivityBySubject = []
    for i in range(len(matrix)):
        tp = matrix[i][i]
        total = sum(matrix[i])
        sensitivityBySubject.append(tp/total)
    sensitivity = np.mean(sensitivityBySubject)*100
    return sensitivity


def calculateSpecificity(matrix):
    specificityBySubject = []
    for s in range(len(matrix)):
        fp = 0
        total = 0
        for i in range(len(matrix)):
            if i != s:
                fp += matrix[i][s]
                total +=sum(matrix[i])
        tn = total-fp
        specificityBySubject.append(tn/total)
    specificity = np.mean(specificityBySubject)*100
    return specificity


def kFoldCrossValidation(dataset, level_of_decomposition, time):
    number_of_folds=10
    classifier_type=[1,2,3,4,5]
    subjects = 32 if dataset == "deap" else 51
    
    amount_classifiers = len(classifier_type)
    final_results = []
    
    accuracy_mean = []
    accuracy_std = []
    detail_accuracies = []
    sensitivity_mean = []
    sensitivity_std = []
    detail_sensitivity = []
    specificity_mean = []
    specificity_std = []
    detail_specificity = []
    
    for i in range(amount_classifiers):
        final_results.append(np.zeros((subjects,subjects)))
        accuracy_mean.append(0)
        accuracy_std.append(0)
        detail_accuracies.append([])
        
        sensitivity_mean.append(0)
        sensitivity_std.append(0)
        detail_sensitivity.append([])
        
        specificity_mean.append(0)
        specificity_std.append(0)
        detail_specificity.append([])
        
    for test in range(number_of_folds):
        X_train, X_test, y_train, y_test = read_file(dataset, level_of_decomposition, time, test)
        for i in range(amount_classifiers):
            results = classification(classifier_type[i], level_of_decomposition, dataset, X_train, y_train, X_test, y_test)  
            final_results[i]+=results
            detail_accuracies[i].append(calculateAccuracy(results))
            detail_sensitivity[i].append(calculateSensitivity(results))
            detail_specificity[i].append(calculateSpecificity(results))
            print(f"Fold-{test} - Classifier-{i} - {time}s - Lvl-{level_of_decomposition} - DONE")
    
    for i in range(amount_classifiers):
        accuracy_mean[i] = np.mean(detail_accuracies[i])
        accuracy_std[i] = np.std(detail_accuracies[i])
        sensitivity_mean[i] = np.mean(detail_sensitivity[i])
        sensitivity_std[i] = np.std(detail_sensitivity[i])
        specificity_mean[i] = np.mean(detail_specificity[i])
        specificity_std[i] = np.std(detail_specificity[i])
    return final_results, detail_accuracies, accuracy_mean, accuracy_std, detail_sensitivity, sensitivity_mean, sensitivity_std, detail_specificity, specificity_mean, specificity_std



def saveData(detailData, level_of_decomposition, name, time):
    for i in range(len(detailData)):
        for j in range(len(detailData[0])):
            detailData[i][j] = str(detailData[i][j]).replace('.',',') 
        
    detailData_data = {'SVM': detailData[0], 'RF': detailData[1], 'KNN': detailData[2], 'AB': detailData[3], 'GNB': detailData[4]}
    detailData_df = pd.DataFrame(data=detailData_data)
    detailData_df.to_excel(f"{argv[1].upper()}/experiment/results/lvl_{level_of_decomposition}/{name}_time_{str(time).replace('.','_')}.xlsx", index=False)
    
    
if __name__ == '__main__':
    if len(argv) != 2:
        print("The correct use is:\npython k_cross.py dataset_name\nDataset names:\n*deap\n*biomex-db")
    else:
        dataset = argv[1]
        if (dataset != "deap" and dataset != "biomex-db"):
            print("Dataset names:\n*deap\n*biomex-db")
        else:
            upperDataset = dataset.upper()
            times = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
            mkdir(f"{upperDataset}/experiment/results")
            for lvl_of_decomposition in range(2,6):
                mkdir(f"{upperDataset}/experiment/results/lvl_{lvl_of_decomposition}")
                    
            clfs = ["SVM", "RF", "KNN", "AB", "GNB"]
            resultsTable_data = {"Time": [], "Classifier": [], "Sensitivity": [], "Specificity": [], "Accuracy": []}
            for level_of_decomposition in range(2, 6):
                for time in times:
                    final_results, detail_accuracies, accuracy_mean, accuracy_std, detail_sensitivity, sensitivity_mean, sensitivity_std, detail_specificity, specificity_mean, specificity_std = kFoldCrossValidation(dataset, level_of_decomposition, time)
                    
                    for i in range(len(clfs)):        
                        resultsTable_data["Time"].append(time)
                        resultsTable_data["Classifier"].append(clfs[i])
                        resultsTable_data["Sensitivity"].append(f"{str(round(sensitivity_mean[i],2)).replace('.',',')}±{str(round(sensitivity_std[i],2)).replace('.',',')}")
                        resultsTable_data["Specificity"].append(f"{str(round(specificity_mean[i],2)).replace('.',',')}±{str(round(specificity_std[i],2)).replace('.',',')}")
                        resultsTable_data["Accuracy"].append(f"{str(round(accuracy_mean[i],2)).replace('.',',')}±{str(round(accuracy_std[i],2)).replace('.',',')}")
                    
                    saveData(detail_accuracies, level_of_decomposition, "accuracy", time)
                    saveData(detail_sensitivity, level_of_decomposition, "sensitivity", time)
                    saveData(detail_specificity, level_of_decomposition, "specificity", time)
                    
                resultsTable = pd.DataFrame(data=resultsTable_data)
                resultsTable.to_excel(f"{upperDataset}/experiment/results/lvl_{level_of_decomposition}/results.xlsx", index=False)
       