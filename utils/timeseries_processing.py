from macro import TIMESTAMPS, PATIENTS, LOCATIONS, WINDOW_SIZE
import data_preprocess
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_reversed_movements_list(movement_lists):
    reversed_movements_list = movement_lists.copy()
    reversed_movements_list.reverse()
    return reversed_movements_list

def get_patients_first_timestamp(movements_list):
    first_timestamp_patient = {}
    for i in tqdm(range(TIMESTAMPS+1)):
        timestamp_movements = list(filter(lambda row: int(row[0])==i, movements_list))
        for ts, patient, ps, bed, bs in timestamp_movements:
            if int(patient) not in first_timestamp_patient:
                first_timestamp_patient[int(patient)] = int(ts)
    return first_timestamp_patient

def get_patients_last_timestamp(reversed_movements_list):
    last_timestamp_patient = {}
    for i in tqdm(sorted(range(TIMESTAMPS+1), reverse=True)):
        timestamp_movements = list(filter(lambda row: int(row[0])==i, reversed_movements_list))
        for ts, patient, ps, bed, bs in timestamp_movements:
            if int(patient) not in last_timestamp_patient:
                last_timestamp_patient[int(patient)] = int(ts)
    return last_timestamp_patient

def get_patients_statistical_info(reversed_movements_list, first_timestamp_patient):
    number_of_comparison = np.zeros(PATIENTS)
    patients_change_status = np.zeros(PATIENTS)
    timestamp_changing = np.zeros(PATIENTS)
    initial_state = np.zeros(PATIENTS)
    for i in tqdm(sorted(range(TIMESTAMPS+1), reverse=True)):
        timestamp_movements = list(filter(lambda row: int(row[0])==i, reversed_movements_list))
        for ts, patient, ps, bed, bs in timestamp_movements:
            number_of_comparison[int(patient)] += 1
            if int(ts) == int(first_timestamp_patient[(int(patient))]):
                initial_state[int(patient)] = int(ps)
            else:
                if int(ps) != initial_state[int(patient)]:
                    initial_state[int(patient)] = int(ps)
                    patients_change_status[int(patient)] = 1
                    timestamp_changing[int(patient)] = ts
    return number_of_comparison, patients_change_status, timestamp_changing, initial_state


def get_patients_to_mantain(number_of_comparison, patients_change_status):
    patients_to_mantain = []
    for idx, i in enumerate(number_of_comparison):
        if i >= WINDOW_SIZE:
            patients_to_mantain.append(int(idx))

    patients_to_mantain_changing_status = []
    for p in np.where(patients_change_status==1)[0]:
        if p in patients_to_mantain:
            patients_to_mantain_changing_status.append(p)
    return patients_to_mantain_changing_status

def get_timestamps_interval(patients_to_mantain_changing_status, first_timestamp_patient, last_timestamp_patient):
    min_ts = TIMESTAMPS
    max_ts = 0
    for p in patients_to_mantain_changing_status:
        min_ts = min_ts if first_timestamp_patient[p] > min_ts else first_timestamp_patient[p]
        max_ts = max_ts if last_timestamp_patient[p] < max_ts else last_timestamp_patient[p]
    return min_ts, max_ts

def get_patients_basic_features(encoded_patients, patients_to_mantain_changing_status):
    patients_basic_features = {}
    for idx,p in enumerate(encoded_patients):
        if idx in patients_to_mantain_changing_status:
            arr = np.array([p[0], p[1]], dtype=int) #p[0] = age, p[1] = sex
            patients_basic_features[idx] = arr
    return patients_basic_features



def get_patients_infos_and_labels(reversed_movements_list, patients_basic_features, min_ts, max_ts, patients_to_mantain_changing_status):
    patients_infos_per_timestamp = {}
    patients_y = {}
    for i in range(min_ts,max_ts+1):
        timestamp_movements = list(filter(lambda row: int(row[0])==i, reversed_movements_list))
        # ts, patient, patients_status, bed, bed_status
        for ts, patient, patients_status, bed, bed_status in timestamp_movements:
            if int(patient) in patients_to_mantain_changing_status:
                basic_info = patients_basic_features[int(patient)]
                if int(patient) not in patients_infos_per_timestamp:
                    other_info = np.array([int(ts) ,int(patients_status), int(bed)])
                    patients_infos_per_timestamp[int(patient)] = [np.concatenate((basic_info, other_info))]
                    patients_y[int(patient)] = [patients_status]
                else:
                    initial_state = patients_infos_per_timestamp[int(patient)][0][3]
                    other_info = np.array([int(ts), initial_state, int(bed)])
                    patients_infos_per_timestamp[int(patient)].append(np.concatenate((basic_info, other_info)))
                    patients_y[int(patient)].append(patients_status)   
    return patients_infos_per_timestamp, patients_y      

def get_batched_data(patients_y, patients_infos_per_timestamp):
    timeseries = []
    y = []
    for i in patients_y.keys():
        for j in range(len(patients_y[i])//15):
            # split the array and add info, y
            timeseries.append(patients_infos_per_timestamp[i][j*15:(j+1)*15])
            y.append(patients_y[i][j*15:(j+1)*15])
        if len(patients_y[i]) % 15 != 0:
            # add the ramaining part to list :-)
            timeseries.append(patients_infos_per_timestamp[i][-15:])
            y.append(patients_y[i][-15:])
    return timeseries, y

def get_data():
    movements_list = data_preprocess.extract_movement_list()
    encoded_patients = data_preprocess.encode_patient_features()
    reversed_movements_list = get_reversed_movements_list(movements_list)
    first_timestamp_patient = get_patients_first_timestamp(movements_list)
    last_timestamp_patient = get_patients_last_timestamp(movements_list)
    number_of_comparison, patients_change_status, _, _ = get_patients_statistical_info(reversed_movements_list, first_timestamp_patient)
    patients_to_mantain_changing_status = get_patients_to_mantain(number_of_comparison, patients_change_status)
    min_ts, max_ts = get_timestamps_interval(patients_to_mantain_changing_status, first_timestamp_patient, last_timestamp_patient)
    patient_basic_features = get_patients_basic_features(encoded_patients, patients_to_mantain_changing_status)
    patients_y, patients_infos_per_timestamp = get_patients_infos_and_labels(reversed_movements_list, patient_basic_features, min_ts, max_ts, patients_to_mantain_changing_status)
    timeseries, y = get_batched_data(patients_y, patients_infos_per_timestamp)
    return timeseries, y 

