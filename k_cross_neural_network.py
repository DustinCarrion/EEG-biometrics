import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from sys import argv


def read_file(dataset, level_of_decomposition, time, number_of_fold): 
    file = f"{dataset.upper()}/experiment/lvl_{level_of_decomposition}/{time}/fold_{number_of_fold}_"
    file_train = f"{file}train.sav"
    file_test = f"{file}test.sav"
    train = joblib.load(file_train)
    test = joblib.load(file_test)
    return train['data'], test['data'], train['labels'], test['labels']


def change_prediction_vector(y_pred):
    y_pred_final = []
    for pred in y_pred:
        y_pred_final.append(np.argmax(pred)+1)
    return y_pred_final


def classification(level_of_decomposition, channels, dataset, subjects, X_train, y_train, X_test, y_test):
    enc = OneHotEncoder(categories="auto")
    y_train = np.array(y_train).reshape(len(y_train), 1)
    y_train = enc.fit_transform(y_train).toarray()

    input_dim = (level_of_decomposition+1)*channels
    model = Sequential()
    if level_of_decomposition == 2:
        if dataset == "deap":
            model.add(Dense(units=106, input_dim=input_dim, kernel_initializer='uniform'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))    
            model.add(Dense(units=subjects, activation='softmax'))
            adam = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=100, epochs=500, verbose=0)
        elif dataset == "biomex-db":
            model.add(Dense(units=127, input_dim=input_dim, kernel_initializer='uniform'))
            model.add(Dropout(0.1))
            model.add(Activation('relu'))    
            model.add(Dense(units=127, kernel_initializer='uniform'))
            model.add(Dropout(0.1))
            model.add(Activation('relu'))    
            model.add(Dense(units=subjects, activation='softmax'))
            adam = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=100, epochs=500, verbose=0)
        
    elif level_of_decomposition == 3:
        if dataset == "deap":
            model.add(Dense(units=106, input_dim=input_dim, kernel_initializer='uniform', kernel_regularizer=l2(0.5)))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
            model.add(Activation('relu'))    
            model.add(Dense(units=subjects, activation='softmax'))
            adam = Adam(learning_rate=0.005)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=100, epochs=100, verbose=0)
        elif dataset == "biomex-db":
            model.add(Dense(units=127, input_dim=input_dim, kernel_initializer='uniform'))
            model.add(Dropout(0.1))
            model.add(Activation('relu'))    
            model.add(Dense(units=127, kernel_initializer='uniform'))
            model.add(Dropout(0.1))
            model.add(Activation('relu'))    
            model.add(Dense(units=subjects, activation='softmax'))
            adam = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=100, epochs=500, verbose=0)
    
    elif level_of_decomposition == 4:
        if dataset == "deap":
            model.add(Dense(units=106, input_dim=input_dim, kernel_initializer='uniform', kernel_regularizer=l2(0.5)))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
            model.add(Activation('relu'))    
            model.add(Dense(units=subjects, activation='softmax'))
            adam = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=100, epochs=500, verbose=0)
        elif dataset == "biomex-db":
            model.add(Dense(units=127, input_dim=input_dim, kernel_initializer='uniform'))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))    
            model.add(Dense(units=127, kernel_initializer='uniform'))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))    
            model.add(Dense(units=subjects, activation='softmax'))
            adam = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=100, epochs=500, verbose=0)
    
    elif level_of_decomposition == 5:
        if dataset == "deap":
            model.add(Dense(units=106, input_dim=input_dim, kernel_initializer='uniform', kernel_regularizer=l2(0.1)))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
            model.add(Activation('relu'))    
            model.add(Dense(units=subjects, activation='softmax'))
            adam = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=100, epochs=500, verbose=0)
        elif dataset == "biomex-db":
            model.add(Dense(units=127, input_dim=input_dim, kernel_initializer='uniform'))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))    
            model.add(Dense(units=127, kernel_initializer='uniform'))
            model.add(Dropout(0.2))
            model.add(Activation('relu'))    
            model.add(Dense(units=subjects, activation='softmax'))
            adam = Adam(learning_rate=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(X_train, y_train, batch_size=100, epochs=500, verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred = change_prediction_vector(y_pred)
    K.clear_session()
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
    if dataset == "deap":
        subjects = 32
        channels = 32
    elif dataset == "biomex-db":
        subjects = 51
        channels = 14

    final_results = np.zeros((subjects,subjects))
    detail_accuracies = []
    detail_sensitivity = []
    detail_specificity = []
    for test in range(number_of_folds):
        X_train, X_test, y_train, y_test = read_file(dataset, level_of_decomposition, time, test)
        results = classification(level_of_decomposition, channels, dataset, subjects, X_train, y_train, X_test, y_test)  
        final_results+=results
        detail_accuracies.append(calculateAccuracy(results))
        detail_sensitivity.append(calculateSensitivity(results))
        detail_specificity.append(calculateSpecificity(results))
        print(f"Fold-{test} - {time}s - lvl-{level_of_decomposition} - DONE")
    
    accuracy_mean = np.mean(detail_accuracies)
    accuracy_std = np.std(detail_accuracies)
    sensitivity_mean = np.mean(detail_sensitivity)
    sensitivity_std = np.std(detail_sensitivity)
    specificity_mean = np.mean(detail_specificity)
    specificity_std = np.std(detail_specificity)
    return final_results, detail_accuracies, accuracy_mean, accuracy_std, detail_sensitivity, sensitivity_mean, sensitivity_std, detail_specificity, specificity_mean, specificity_std



def saveData(detailData, level_of_decomposition, name, time):
    for i in range(len(detailData)):
        detailData[i] = str(detailData[i]).replace('.',',') 
    
    detailData_data = {'MLP': detailData}
    detailData_df = pd.DataFrame(data=detailData_data)
    detailData_df.to_excel(f"{argv[1].upper()}/experiment/results/lvl_{level_of_decomposition}/{name}_mlp_time_{str(time).replace('.','_')}.xlsx", index=False)


if __name__ == '__main__':
    if len(argv) != 2:
        print("The correct use is:\npython k_cross_neural_network.py dataset_name\nDataset names:\n*deap\n*biomex-db")
    else:
        dataset = argv[1]
        if (dataset != "deap" and dataset != "biomex-db"):
            print("Dataset names:\n*deap\n*biomex-db")
        else:
            times = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
            resultsTable_data = {"Time": [], "Classifier": [], "Sensitivity": [], "Specificity": [], "Accuracy": []}
            for level_of_decomposition in range(2, 6):
                for time in times:
                    final_results, detail_accuracies, accuracy_mean, accuracy_std, detail_sensitivity, sensitivity_mean, sensitivity_std, detail_specificity, specificity_mean, specificity_std = kFoldCrossValidation(dataset, level_of_decomposition, time)
                    
                    resultsTable_data["Time"].append(time)
                    resultsTable_data["Classifier"].append("MLP")
                    resultsTable_data["Sensitivity"].append(f"{str(round(sensitivity_mean,2)).replace('.',',')}±{str(round(sensitivity_std,2)).replace('.',',')}")
                    resultsTable_data["Specificity"].append(f"{str(round(specificity_mean,2)).replace('.',',')}±{str(round(specificity_std,2)).replace('.',',')}")
                    resultsTable_data["Accuracy"].append(f"{str(round(accuracy_mean,2)).replace('.',',')}±{str(round(accuracy_std,2)).replace('.',',')}")
                    
                    saveData(detail_accuracies, level_of_decomposition, "accuracy", time)
                    saveData(detail_sensitivity, level_of_decomposition, "sensitivity", time)
                    saveData(detail_specificity, level_of_decomposition, "specificity", time)
                 
                resultsTable = pd.DataFrame(data=resultsTable_data)
                resultsTable.to_excel(f"{dataset.upeer()}/experiment/results/lvl_{level_of_decomposition}/results_mlp.xlsx", index=False)
