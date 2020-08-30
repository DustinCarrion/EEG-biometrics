import pickle
import numpy as np
from pywt import Wavelet
from math import floor, ceil
import joblib
from sys import argv
from os import mkdir

def downsampling(signal,factor = 8):
    ds_signal = np.array([])
    for i in range(0,len(signal),factor):
        ds_signal = np.append(ds_signal,signal[i])
    return ds_signal


def padding_symmetric(signal):
    return np.concatenate([np.flipud(signal[:8]),signal,np.flipud(signal[-8:])])


def upsampling(signal,reconstruction_filter,real_len):
    restored_signal = np.zeros(2 * len(signal) + 1)
    for i in range(len(signal)):
        restored_signal[i*2+1] = signal[i]
    restored_signal = np.convolve(restored_signal, reconstruction_filter)
    restored_len = len(restored_signal)
    exceed_len = (restored_len - real_len) / 2.0
    restored_signal = restored_signal[int(floor(exceed_len)):(restored_len - int(ceil(exceed_len)))]
    return restored_signal


def wavelet_decomposition(signal,level):
    original_len = len(signal)
    A_coeff = []
    D_coeff = []
    wavelet = Wavelet('db4')
    Lo_D = wavelet.dec_lo
    Hi_D = wavelet.dec_hi
    for i in range(level):
        padding_signal = padding_symmetric(signal)
        a_signal = np.convolve(padding_signal,Lo_D)[8:8+len(signal)+8-1]
        a_signal = a_signal[1:len(a_signal):2]
        d_signal = np.convolve(padding_signal,Hi_D)[8:8+len(signal)+8-1]
        d_signal = d_signal[1:len(d_signal):2]
        A_coeff.append(a_signal)
        D_coeff.append(d_signal)
        signal = a_signal
    Lo_R = wavelet.rec_lo
    Hi_R = wavelet.rec_hi
    real_lens = []
    for i in range(level-2,-1,-1):
        real_lens.append(len(A_coeff[i]))
    real_lens.append(original_len)
    restored_A_coeff = []
    for i in range(level):
        restored_A = upsampling(A_coeff[i], Lo_R, real_lens[(level-1)-i])
        for j in range(i):
            restored_A = upsampling(restored_A, Lo_R, real_lens[level-i+j])
        restored_A_coeff.append(restored_A)
    restored_D_coeff = []
    for i in range(level):
        restored_D = upsampling(D_coeff[i], Hi_R, real_lens[(level-1)-i])
        for j in range(i):
            restored_D = upsampling(restored_D, Lo_R, real_lens[level-i+j])
        restored_D_coeff.append(restored_D)
    return restored_A_coeff,restored_D_coeff 


def energy(vector): 
    value = 0
    for i in range(0,len(vector)):
        value += vector[i]**2 
    return value
    

def writeFileTimeSegmentation(dataset, sec, input_data_path, dwt_lvl, output_data_path):
    frequency = 128
    if dataset == "deap":
        subjects = 32
        total_recording_time = 60
        baseline = 3
    elif dataset == "biomex-db":
        subjects = 51
        total_recording_time = 2.5
        baseline = 0
        
        
    time = int(frequency*sec)
    for i in range(1,subjects+1):
        if dataset == "deap":
            name = f"{input_data_path}/s0{i}.dat" if len(str(i)) == 1 else f"{input_data_path}/s{i}.dat" 
            file = open(name, 'rb') 
            dictionary = pickle.load(file, encoding='bytes') 
            data = dictionary[b'data']
        elif dataset == "biomex-db":
            name = f"{input_data_path}/s0{i}.sav" if len(str(i)) == 1 else f"{input_data_path}/s{i}.sav"
            file = joblib.load(name)
            data = file['data']
        segmented_data = []
        
        for video in data:
            segmented_video = []
            start = np.random.randint(0,((total_recording_time*frequency)-time)+1) 
            for channel in video:
                segmented_video.append(channel[(frequency*baseline)+start:(frequency*baseline)+start+time]) 
            segmented_data.append(segmented_video)
        
        rwe_matrix = []
        for video in segmented_data:
            rwe_vector = []
            for channel in video:
                A,D = wavelet_decomposition(channel, dwt_lvl)
                energies = [energy(A[-1])]
                for Di in D:
                    energies.append(energy(Di))
                total_energy = sum(energies)
                for energyCoeff in energies:
                    rwe_vector.append(energyCoeff/total_energy)
            rwe_matrix.append(rwe_vector)
        
        joblib.dump(rwe_matrix, f"{output_data_path}/lvl_{dwt_lvl}/{sec}/s0{i}.sav" if len(str(i)) == 1 else f"{output_data_path}/lvl_{dwt_lvl}/{sec}/s{i}.sav")
        print(f"Lvl-{dwt_lvl} - Subject {i} - {sec}s - DONE")


if __name__ == '__main__':
    if len(argv) != 3:
        print("The correct use is:\npython create_feature_matrices.py input_data_path dataset_name\nDataset names:\n*deap\n*biomex-db")
    else:
        input_data_path = argv[1]
        dataset = argv[2]
        if (dataset != "deap" and dataset != "biomex-db"):
            print("Dataset names:\n*deap\n*biomex-db")
        else:
            data_dir = "DEAP/data" if dataset=="deap" else "BIOMEX-DB/data"
            mkdir(data_dir)
            output_data_path = f"{data_dir}/feature_matrices"
            mkdir(output_data_path)
            times = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
            try:
                for dwt_lvl in range(2,6):
                    mkdir(f"{output_data_path}/lvl_{dwt_lvl}")
                    for time in times:
                        mkdir(f"{output_data_path}/lvl_{dwt_lvl}/{time}")
                        writeFileTimeSegmentation(dataset, time, input_data_path, dwt_lvl, output_data_path)
            except:
                print("Error")