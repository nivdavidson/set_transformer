import pickle
import numpy as np
Y_TRAIN = 'y_train'
Y_TEST = 'y_test'
Y_VAL = 'y_val'
X_TRAIN = 'x_train'
X_TEST = 'x_test'
X_VAL = 'x_val'

def get_data(data_file_path, fields=["px", "py", "pz"]):
    with open(data_file_path, 'rb') as f:
        data = pickle.load(f)

    formatted_data = {
        Y_TRAIN : data[Y_TRAIN].copy(),
        Y_TEST: data[Y_TEST].copy(),
        Y_VAL: data[Y_VAL].copy()
    }

    particle_id_fields = ['kaonID', 'pionID', 'electronID', 'muonID', 'protonID']
    track_fields = data['fields']['track']
    gamma_fields = data['fields']['gamma']
    btag_fields = data['fields']['Btag']

    common_values = list(set(track_fields) & set(gamma_fields) & set(btag_fields))
    for field in fields:
        if field not in common_values:
            print(f'field {field} is not valid')
            return

    for label in [X_TRAIN, X_TEST, X_VAL]:
        x_vals = data[label]
        new_vals= []
        for event in x_vals:
            new_event = []
            for track in event['track']:
                new_track = []
                for field in fields:
                    new_track.append(track[track_fields.index(field)])
                for field in particle_id_fields:
                    new_track.append(track[track_fields.index(field)])
                # GammaID, BtagID
                new_track.extend([0, 0])
                new_event.append(np.array(new_track, dtype=float))
            for gamma in event['gamma']:
                new_gamma = []
                for field in fields:
                    new_gamma.append(gamma[gamma_fields.index(field)])
                new_gamma.extend([0 for i in range(len(particle_id_fields))])
                # GammaID, BtagID
                new_gamma.extend([1, 0])
                new_event.append(np.array(new_gamma, dtype=np.float))
            for btag in event['Btag']:
                new_btag = []
                for field in fields:
                    new_btag.append(btag[btag_fields.index(field)])
                new_btag.extend([0 for i in range(len(particle_id_fields))])
                # GammaID, BtagID
                new_btag.extend([0, 1])
                new_event.append(np.array(new_btag, dtype=np.float))
            new_vals.append(new_event)
        formatted_data[label] = np.array(new_vals, dtype=object)
    return formatted_data
