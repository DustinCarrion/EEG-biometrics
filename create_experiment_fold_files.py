import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sys import argv
from os import mkdir


def read_file(file_name, channels, level_of_decomposition): 
    aux = joblib.load(file_name)
    dataMatrix = []
    for i in aux:
        vector = []
        for j in range(0,channels*(level_of_decomposition+1)): #32 = numero de canales de EEG
            vector.append(i[j])
        dataMatrix.append(vector)
    return dataMatrix


def trainTestSplit(dataset, level_of_decomposition, time, files, channels, train_files, test_files):
    directory = f"{dataset}/data/experiments/lvl_{level_of_decomposition}/{time}"
    X = []
    y = []
    counter = 0
    for file in files:
        counter += 1
        data = read_file(f"{directory}/{file}", channels, level_of_decomposition)
        X.append(data)
        y.append([counter]*len(data))
    X_train_ordered = []
    X_test_ordered = []
    y_train_ordered = []
    y_test_ordered = []
    for i in range(len(files)):
        X_train_aux = [] 
        y_train_aux = []
        for j in train_files[i]:
            X_train_aux.append(X[i][j])
            y_train_aux.append(y[i][j])
        
        X_test_aux = [] 
        y_test_aux = []
        for j in test_files[i]:
            X_test_aux.append(X[i][j])
            y_test_aux.append(y[i][j])
            
        X_train_ordered += X_train_aux
        X_test_ordered += X_test_aux
        y_train_ordered += y_train_aux
        y_test_ordered += y_test_aux
    
    train_shuffle = np.random.choice(len(X_train_ordered),len(X_train_ordered),replace=False)
    X_train = []
    y_train = []
    for i in train_shuffle:
        X_train.append(X_train_ordered[i])
        y_train.append(y_train_ordered[i])
    test_shuffle = np.random.choice(len(X_test_ordered),len(X_test_ordered),replace=False)
    X_test = []
    y_test= []
    for i in test_shuffle:
        X_test.append(X_test_ordered[i])
        y_test.append(y_test_ordered[i])
    
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    train = {'data':X_train, 'labels': y_train}
    test = {'data':X_test, 'labels': y_test}
    return train, test

def createFoldFiles(dataset):
    number_of_folds = 10
    train_size = 0.75
    times = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    if dataset == "DEAP":
        subjects = 32
        channels = 32
        trials_per_subject = 32
    elif dataset == "BIOMEX-DB":
        subjects = 51
        channels = 14
        trials_per_subject = 108
    files = [f"s0{subject}.sav" if len(str(subject)) == 1 else f"s{subject}.sav" for subject in range(1,subjects+1)]
    mkdir(f"{dataset}/experiment")
    for lvl_of_decomposition in range(2,6):
        mkdir(f"{dataset}/experiment/lvl_{lvl_of_decomposition}")
        for time in times:
            mkdir(f"{dataset}/experiment/lvl_{lvl_of_decomposition}/{time}")
    
    for i in range(number_of_folds):
        train_files = []
        for j in range(subjects): 
            train_files.append(np.random.choice(trials_per_subject, int(train_size*trials_per_subject), replace=False))
        
        test_files = []
        for j in range(subjects):
            indexes = []
            for k in range(trials_per_subject):
                if k not in train_files[j]:
                    indexes.append(k)
            np.random.shuffle(indexes)
            test_files.append(indexes)
        
        for lvl_of_decomposition in range(2,6):
            for time in times:
                train, test = trainTestSplit(dataset, lvl_of_decomposition, time, files, channels, train_files, test_files)
                joblib.dump(train, f"{dataset}/experiment/lvl_{lvl_of_decomposition}/{time}/fold_{i}_train.sav")
                joblib.dump(test, f"{dataset}/experiment/lvl_{lvl_of_decomposition}/{time}/fold_{i}_test.sav")
                print(f"Fold-{i} - lvl-{lvl_of_decomposition} - {time}s - DONE")

if __name__ == '__main__':    
    if len(argv) != 2:
        print("The correct use is:\npython create_experiment_fold_files.py dataset_name\nDataset names:\n*deap\n*biomex-db")
    else:
        dataset = argv[1]
        if (dataset != "deap" and dataset != "biomex-db"):
            print("Dataset names:\n*deap\n*biomex-db")
        else:
            createFoldFiles(dataset.upper())
