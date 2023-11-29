import pandas as pd
import numpy as np

TIMESTAMPS = 1094
PATIENTS = 6962
LOCATIONS = 340

patients = pd.read_csv('/home/cc/tgn/umu/patients.csv', sep=";",header=None, names=['id' , 'age' , 'sex' , 'LOS' , 'incubation_time' , 'infection_time' , 'treatment_days' , 'treatment' , 'antibiotic' , 'microorganism' , 'adm_day' , 'last_day' , 'died' , 'LOS_final', 'colonized', 'non_susceptible'])
movements = pd.read_csv('/home/cc/tgn/umu/movements.csv',header=None)
location = pd.read_csv('/home/cc/tgn/umu/Locations.csv')

encoded_sex = lambda sex : 1 if sex=='M' else 0
encoded_ns = lambda ns : 1 if ns==True else 0
encoded_bed_status = lambda bool_str : 1 if bool_str=='True' else 0


def extract_location_features():
    return location[['id_location','infected']].sort_values(by='id_location')

def extract_patient_features():
    return patients[['id', 'age', 'sex', 'non_susceptible']]

def extract_movement_list():
    movements_list = []
    for idx,row in movements.iterrows():
        m = row[0].split(';')
        timestamp = m[0]
        for x in batch(m[1:], 4):
            if len(x) < 4:
                continue
            if 'places' in x:
                continue
            movements_list.append((timestamp, x[0], x[1], x[2], x[3]))
    to_remove = list(filter(lambda row: int(row[2]) > 5, movements_list))
    for x in to_remove:
        movements_list.remove(x)
    return movements_list

def generate_links():
    ids_to_name = {}
    for idx, row in location.iterrows():
        ids_to_name[row['id_location']] = row['name']

    links = []
    not_added = 0
    for idx, row in location.loc[location['id_parent'].isnull()==False].iterrows():
        if int(row['id_parent']) in ids_to_name:
            links.append((int(row['id_parent']),int(row['id_location'])))
        else:
            not_added += 1
    return links, not_added

def generate_edges_sequences(links, movements_list):
    last_known_position = {}
    for i in range(PATIENTS + 1):
        last_known_position[i] = None
    patients_locations_l = []
    weights_l = []
    for i in range(0,TIMESTAMPS +1):
        updated_patients = []
        data = {}
        weights = {}
        l = list(filter(lambda row: int(row[0])==i, movements_list))
        actual_edges = []
        reversed_edges = []
        for ts, patient, patient_status, bed, bed_status in l:
            updated_patients.append(int(patient))
            last_known_position[int(patient)] = int(bed)
            actual_edges.append((int(patient),int(bed)))
            reversed_edges.append((int(bed),int(patient)))
        for j in range(PATIENTS ):
            if j not in updated_patients and last_known_position[j] is not None:
                actual_edges.append((j, last_known_position[j]))
        
        data[("patient","stay","location")] = np.array(actual_edges)
        data[("location","hierarchy","location")] = np.array(links)
        data[("location","has","patient")] = np.array(reversed_edges)
        weights[("patient","stay","location")] = np.ones(len(actual_edges))
        weights[("location","hierarchy","location")] = np.ones(len(links))
        weights[("location","has","patient")] = np.ones(len(reversed_edges))
        patients_locations_l.append(data)
        weights_l.append(weights)

    return patients_locations_l, weights_l


def generate_nodes_sequences(movements_list, encoded_patients):
    location_status = location.sort_values(by='id_location')[['id_location','infected']].values.tolist()
    last_known_location_status = {}
    for row in location_status:
        last_known_location_status[row[0]] = row[1]
    location_status_l = []
    for i in range(0,TIMESTAMPS +1):
        updated_location = []
        data = {}
        l = list(filter(lambda row: int(row[0])==i, movements_list))
        actual_status = np.zeros(LOCATIONS)
        for ts, patient, patient_status, bed, bed_status in l:

            actual_status[int(bed)] = encoded_bed_status(bed_status)

        
        data["location"] = np.array(actual_status)
        data["patient"] = np.array(encoded_patients)
        location_status_l.append(data)
    return location_status_l

def generate_patients_labels(movements_list):
    patient_state_l = []
    actual_status = np.zeros(PATIENTS)
    for i in range(0,TIMESTAMPS +1):
        data = {}
        l = list(filter(lambda row: int(row[0])==i, movements_list))
        
        for ts, patient, patient_status, bed, bed_status in l:
            actual_status[int(patient)] = int(patient_status)

        
        data[("patient")] = np.array(actual_status)
        patient_state_l.append(data)
    return patient_state_l
    

def encode_patient_features():
    patient_features = extract_patient_features()
    patients_feature_list = patient_features[['age','sex','non_susceptible']].values.tolist()
    encoded_patients = list(map(lambda row: [row[0], encoded_sex(row[1]), encoded_ns(row[2])], patients_feature_list))
    return encoded_patients

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_data():
    links, not_added = generate_links()
    movement_list = extract_movement_list()
    patients_locations_l, weights_l = generate_edges_sequences(links,movement_list)
    encoded_patients = encode_patient_features()
    location_status_l = generate_nodes_sequences(movement_list, encoded_patients)
    patient_state_l = generate_patients_labels(movement_list)
    return patients_locations_l, weights_l, location_status_l, patient_state_l