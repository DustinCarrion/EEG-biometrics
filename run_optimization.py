import os
os.system("python create_optimization_fold_files.py deap")
os.system("python create_optimization_fold_files.py biomex-db")
os.system("python grid_search.py deap")
os.system("python grid_search.py biomex-db")
os.system("python grid_search_neural_network.py deap")
os.system("python grid_search_neural_network.py biomex-db")
os.system("python read_optimization_results.py deap")
os.system("python read_optimization_results.py biomex-db")