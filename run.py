import os
os.system("python split_feature_matrices.py deap")
os.system("python split_feature_matrices.py biomex-db")
os.system("python run_experiment.py")
