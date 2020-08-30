import joblib
import numpy as np
from sys import argv
from os import mkdir
    
def saveSplitData(input_data_path, data_path, decomposition_lvl, time, subject, optimization_index_per_subject, experiment_index_per_subject):
    name = f"{input_data_path}/lvl_{decomposition_lvl}/{time}/s0{subject}.sav" if len(str(subject)) == 1 else f"{input_data_path}/lvl_{decomposition_lvl}/{time}/s{subject}.sav"
    data_matrix = joblib.load(name)
    
    if time == 1.25:
        optimization_matrix = []
        for eeg in optimization_index_per_subject[i-1]:
            optimization_matrix.append(data_matrix[eeg]) 
        name_optimization = f"{data_path}/optimization/lvl_{decomposition_lvl}/s0{subject}.sav" if len(str(subject)) == 1 else f"{data_path}/optimization/lvl_{decomposition_lvl}/s{subject}.sav"            
        joblib.dump(optimization_matrix, name_optimization)
    
    experiment_matrix = []
    for eeg in experiment_index_per_subject[i-1]:
        experiment_matrix.append(data_matrix[eeg])
    name_experiment = f"{data_path}/experiments/lvl_{decomposition_lvl}/{time}/s0{subject}.sav" if len(str(subject)) == 1 else f"{data_path}/experiments/lvl_{decomposition_lvl}/{time}/s{subject}.sav"
    joblib.dump(experiment_matrix, name_experiment)
    
    print(f"Lvl-{decomposition_lvl} - Subject {subject} - {time}s - DONE")
    
    
if __name__ == '__main__':
    if len(argv) != 2:
        print("The correct use is:\npython split_feature_matrices.py dataset_name\nDataset names:\n*deap\n*biomex-db")
    else:
        dataset = argv[1]
        if (dataset != "deap" and dataset != "biomex-db"):
            print("Dataset names:\n*deap\n*biomex-db")
        else:
            percentage_for_optimization = 0.2
            if dataset == "deap":
                trials_per_subject = 40
                number_of_subjects = 32
                data_path = "DEAP/data"
            elif dataset == "biomex-db":
                trials_per_subject = 135
                number_of_subjects = 51
                data_path = "BIOMEX-DB/data"
                
            input_data_path = f"{data_path}/feature_matrices"
            experiment_data_path = f"{data_path}/experiments"
            optimization_data_path = f"{data_path}/optimization"
            mkdir(experiment_data_path)
            mkdir(optimization_data_path)
            
            optimization_index_per_subject = []
            for i in range(number_of_subjects): 
                optimization_index_per_subject.append(np.random.choice(trials_per_subject, int(percentage_for_optimization*trials_per_subject), replace=False))
            
            experiment_index_per_subject = []
            for i in range(number_of_subjects):
                indexes = []
                for j in range(trials_per_subject):
                    if j not in optimization_index_per_subject[i]:
                        indexes.append(j)
                experiment_index_per_subject.append(indexes)
            
            times = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
            for dwt_lvl in range(2,6):
                mkdir(f"{experiment_data_path}/lvl_{dwt_lvl}")
                mkdir(f"{optimization_data_path}/lvl_{dwt_lvl}")
                for time in times:
                    mkdir(f"{experiment_data_path}/lvl_{dwt_lvl}/{time}")
                    for subject in range(1,number_of_subjects+1):
                        saveSplitData(input_data_path, data_path, dwt_lvl, time, subject, optimization_index_per_subject, experiment_index_per_subject)
                    