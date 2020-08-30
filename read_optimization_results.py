import joblib
from sys import argv

def printResults(lvl, clf):
    dataset = argv[1].upper()
    acc = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{lvl}/acc_{clf}.sav")
    parameters = joblib.load(f"{dataset}/hyperparameter_optimization/results/lvl_{lvl}/parameters_{clf}.sav")
    print(f"\n\n------------ RESULTS {lvl}/{clf} ------------")
    best_acc = max(acc)
    best_param = parameters[acc.index(best_acc)]
    print(best_param)
        

if __name__ == "__main__":
    if len(argv) != 2:
        print("The correct use is:\npython read_optimization_results.py dataset_name\nDataset names:\n*deap\n*biomex-db")
    else:
        dataset = argv[1]
        if (dataset != "deap" and dataset != "biomex-db"):
            print("Dataset names:\n*deap\n*biomex-db")
        else:
            for lvl in range(2,6):
                printResults(lvl, "svm")
    
            for lvl in range(2,6):
                printResults(lvl, "rf")
                
            for lvl in range(2,6):
                printResults(lvl, "knn")
                
            for lvl in range(2,6):
                printResults(lvl, "ab")
                
            for lvl in range(2,6):
                printResults(lvl, "gnb")
                
            for lvl in range(2,6):
                printResults(lvl, "mlp")