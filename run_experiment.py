import os
os.system("python create_experiment_fold_files.py deap")
os.system("python create_experiment_fold_files.py biomex-db")
os.system("python k_cross.py deap")
os.system("python k_cross.py biomex-db")
os.system("python k_cross_neural_network.py deap")
os.system("python k_cross_neural_network.py biomex-db")
