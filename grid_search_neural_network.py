import gc
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from sys import argv


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


def change_prediction_vector(y_pred):
    y_pred_final = []
    for pred in y_pred:
        y_pred_final.append(np.argmax(pred)+1)
    return y_pred_final


def checkParameters(parametersToSave, savedParameters):
    if parametersToSave in savedParameters:
        return True
    return False


if __name__ == '__main__':
    if len(argv) != 2:
        print("The correct use is:\npython grid_search_neural_network.py dataset_name\nDataset names:\n*deap\n*biomex-db")
    else:
        dataset = argv[1]
        if (dataset != "deap" and dataset != "biomex-db"):
            print("Dataset names:\n*deap\n*biomex-db")
        else:
            if dataset == "deap":
                subjects = 32
                channels = 32
            elif dataset == "biomex-db":
                subjects = 51
                channels = 14
            dataset = dataset.upper()
            
            net_specifications = [[106], [106,106], [106,106,106], [84,84], [127,127]] 
            learning_rate = [1e-3, 5e-3, 0.01, 0.05, 0.1] 
            batch_normalization = [True, False] 
            dropout = [True, False] 
            dropout_percentage = [0.1, 0.2, 0.3, 0.4, 0.5] 
            l2_regularization = [True, False] 
            l2_regularization_values = [0.01, 0.05, 0.1, 0.5] 
            epochs = [10, 100, 500] 
            
            for decomposition_level in range(2,6):
                input_dim = (decomposition_level+1)*channels
                try:
                    acc = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_mlp.sav")
                    parameters = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_mlp.sav")
                except:
                    acc = [] 
                    parameters = []
                    
                for i in range(len(net_specifications)):
                    for j in range(len(learning_rate)):
                        for k in range(len(batch_normalization)):
                            for l in range(len(dropout)):
                                for m in range(len(l2_regularization)):
                                    for n in range(len(epochs)):
                                        
                                        if dropout[l] and l2_regularization[m]:
                                            for o in range(len(dropout_percentage)):                          
                                                for p in range(len(l2_regularization_values)):
                                                    if checkParameters([net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], dropout_percentage[o], l2_regularization[m], l2_regularization_values[p], epochs[n]], parameters):
                                                        continue
                                                    model = Sequential()
                                                    for q in range(len(net_specifications[i])):
                                                        if q == 0:
                                                            model.add(Dense(units = net_specifications[i][q], input_dim=input_dim, kernel_initializer = 'uniform'))
                                                        else:
                                                            model.add(Dense(units= net_specifications[i][q], kernel_regularizer=l2(l2_regularization_values[p])))
                                                        if batch_normalization[k]:
                                                            model.add(BatchNormalization())
                                                        model.add(Dropout(dropout_percentage[o]))
                                                        model.add(Activation('relu'))  
                                                    model.add(Dense(units=subjects, activation='softmax'))
                                                    adam = Adam(learning_rate=learning_rate[j])
                                                    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                                                    
                                                    accuracies = []
                                                    for fold in range(10):
                                                        X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                                                        enc = OneHotEncoder(categories="auto")
                                                        y_train = np.array(y_train).reshape(len(y_train), 1)
                                                        y_train = enc.fit_transform(y_train).toarray()
                                                        model.fit(X_train, y_train, batch_size = 100, epochs = epochs[n], verbose=0)
                                                        y_pred = model.predict(X_test) 
                                                        y_pred = change_prediction_vector(y_pred)
                                                        accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                    
                                                    acc.append(np.mean(accuracies))
                                                    parameters.append([net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], dropout_percentage[o], l2_regularization[m], l2_regularization_values[p], epochs[n]])
                                                    print(net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], dropout_percentage[o], l2_regularization[m], l2_regularization_values[p], epochs[n])
                                                    del model, adam, accuracies, X_train, X_test, y_train, y_test, enc, y_pred
                                                    K.clear_session()
                                                    gc.collect()
                                                    joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_mlp.sav")
                                                    joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_mlp.sav")
                                                      
                                        elif dropout[l]:
                                            for o in range(len(dropout_percentage)):  
                                                if checkParameters([net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], dropout_percentage[o], l2_regularization[m], epochs[n]], parameters):
                                                    continue
                                                model = Sequential()
                                                for p in range(len(net_specifications[i])):     
                                                    if p == 0:
                                                        model.add(Dense(units = net_specifications[i][p], input_dim=input_dim, kernel_initializer = 'uniform'))
                                                    else:
                                                        model.add(Dense(units= net_specifications[i][p]))
                                                    if batch_normalization[k]:
                                                        model.add(BatchNormalization())
                                                    model.add(Dropout(dropout_percentage[o]))
                                                    model.add(Activation('relu'))  
                                                model.add(Dense(units=subjects, activation='softmax'))
                                                adam = Adam(learning_rate=learning_rate[j])
                                                model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                                                    
                                                accuracies = []
                                                for fold in range(10):
                                                    X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                                                    enc = OneHotEncoder(categories="auto")
                                                    y_train = np.array(y_train).reshape(len(y_train), 1)
                                                    y_train = enc.fit_transform(y_train).toarray()
                                                    model.fit(X_train, y_train, batch_size = 100, epochs = epochs[n], verbose=0)
                                                    y_pred = model.predict(X_test) 
                                                    y_pred = change_prediction_vector(y_pred)
                                                    accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                                                acc.append(np.mean(accuracies))
                                                parameters.append([net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], dropout_percentage[o], l2_regularization[m], epochs[n]])
                                                print(net_specifications[i],learning_rate[j],batch_normalization[k],dropout[l],dropout_percentage[o],l2_regularization[m],epochs[n])
                                                del model, adam, accuracies, X_train, X_test, y_train, y_test, enc, y_pred
                                                K.clear_session()
                                                gc.collect()
                                                joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_mlp.sav")
                                                joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_mlp.sav")
                                                
                                                         
                                        elif l2_regularization[m]:                         
                                            for o in range(len(l2_regularization_values)):
                                                if checkParameters([net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], l2_regularization[m], l2_regularization_values[o], epochs[n]], parameters):
                                                    continue
                                                model = Sequential()
                                                for p in range(len(net_specifications[i])):
                                                    if p == 0:
                                                        model.add(Dense(units = net_specifications[i][p], input_dim=input_dim, kernel_initializer = 'uniform'))
                                                    else:
                                                        model.add(Dense(units= net_specifications[i][p], kernel_regularizer=l2(l2_regularization_values[o])))
                                                    if batch_normalization[k]:
                                                        model.add(BatchNormalization())
                                                    model.add(Dropout(dropout_percentage[o]))
                                                    model.add(Activation('relu'))  
                                                model.add(Dense(units=subjects, activation='softmax'))
                                                adam = Adam(learning_rate=learning_rate[j])
                                                model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                                                
                                                accuracies = []
                                                for fold in range(10):
                                                    X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                                                    enc = OneHotEncoder(categories="auto")
                                                    y_train = np.array(y_train).reshape(len(y_train), 1)
                                                    y_train = enc.fit_transform(y_train).toarray()
                                                    model.fit(X_train, y_train, batch_size = 100, epochs = epochs[n], verbose=0)
                                                    y_pred = model.predict(X_test) 
                                                    y_pred = change_prediction_vector(y_pred)
                                                    accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                                                acc.append(np.mean(accuracies))
                                                parameters.append([net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], l2_regularization[m], l2_regularization_values[o], epochs[n]])
                                                print(net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], l2_regularization[m], l2_regularization_values[o], epochs[n])
                                                del model, adam, accuracies, X_train, X_test, y_train, y_test, enc, y_pred
                                                K.clear_session()
                                                gc.collect()
                                                joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_mlp.sav")
                                                joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_mlp.sav")
                                                                       
                                        else:
                                            if checkParameters([net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], l2_regularization[m], epochs[n]], parameters):
                                                continue
                                            model = Sequential()
                                            for o in range(len(net_specifications[i])):
                                                if o == 0:
                                                    model.add(Dense(units = net_specifications[i][o], input_dim=input_dim, kernel_initializer = 'uniform'))
                                                else:
                                                    model.add(Dense(units= net_specifications[i][o]))
                                                if batch_normalization[k]:
                                                    model.add(BatchNormalization())
                                                model.add(Activation('relu'))  
                                            model.add(Dense(units=subjects, activation='softmax'))
                                            adam = Adam(learning_rate=learning_rate[j])
                                            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                                            
                                            accuracies = []
                                            for fold in range(10):
                                                X_train, X_test, y_train, y_test = read_file(fold, decomposition_level)
                                                enc = OneHotEncoder(categories="auto")
                                                y_train = np.array(y_train).reshape(len(y_train), 1)
                                                y_train = enc.fit_transform(y_train).toarray()
                                                model.fit(X_train, y_train, batch_size = 100, epochs = epochs[n], verbose=0)
                                                y_pred = model.predict(X_test) 
                                                y_pred = change_prediction_vector(y_pred)
                                                accuracies.append(calculateAccuracy(confusion_matrix(y_test,y_pred)))
                                            acc.append(np.mean(accuracies))
                                            parameters.append([net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], l2_regularization[m], epochs[n]])
                                            print(net_specifications[i], learning_rate[j], batch_normalization[k], dropout[l], l2_regularization[m], epochs[n])
                                            del model, adam, accuracies, X_train, X_test, y_train, y_test, enc, y_pred
                                            K.clear_session()
                                            gc.collect()
                                            joblib.dump(acc, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/acc_mlp.sav")
                                            joblib.dump(parameters, f"{dataset}/hyperparameter_optimization/results/lvl_{decomposition_level}/parameters_mlp.sav")
                                            
                print(f"lvl-{decomposition_level} - DONE")
