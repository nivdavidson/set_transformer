import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

Y_TRAIN = 'y_train'
Y_TEST = 'y_test'
Y_VAL = 'y_val'
X_TRAIN = 'x_train'
X_TEST = 'x_test'
X_VAL = 'x_val'

BATCH_SIZE = 64


class CombinedDataset(Dataset):
    def __init__(self, signal_data, background_data, x_label, y_label):
        self.x = signal_data[x_label] + background_data[x_label]
        self.y = np.concatenate([signal_data[y_label], background_data[y_label]])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dataloaders(signal_data_file_path, background_data_file_path, batch_size=BATCH_SIZE):
    signal_data = get_data(signal_data_file_path)
    background_data = get_data(background_data_file_path)

    dataloaders = []
    for x_key, y_key in ((X_TRAIN, Y_TRAIN),
                         (X_VAL, Y_VAL),
                         (X_TEST, Y_TEST)):
        dataset = CombinedDataset(signal_data, background_data, x_key, y_key)
        dataloaders.append(DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=lambda batch: batch))
    return dataloaders


def get_data(data_file_path, fields=("px", "py", "pz")):
    with open(data_file_path, 'rb') as f:
        data = pickle.load(f)

    formatted_data = {
        Y_TRAIN: data[Y_TRAIN].copy(),
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
        # ### padding ###
        # max_events = 0
        # for event in x_vals:
        #     if len(event['track']) + len(event['gamma']) > max_events:
        #         max_events = len(event['track']) + len(event['gamma'])
        # max_events = max_events + 1
        # ### end of padding ###
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
                new_event.append(np.array(new_track, dtype=np.double))
            for gamma in event['gamma']:
                new_gamma = []
                for field in fields:
                    new_gamma.append(gamma[gamma_fields.index(field)])
                new_gamma.extend([0 for i in range(len(particle_id_fields))])
                # GammaID, BtagID
                new_gamma.extend([1, 0])
                new_event.append(np.array(new_gamma, dtype=np.double))
            for btag in event['Btag']:
                new_btag = []
                for field in fields:
                    new_btag.append(btag[btag_fields.index(field)])
                new_btag.extend([0 for i in range(len(particle_id_fields))])
                # GammaID, BtagID
                new_btag.extend([0, 1])
                new_event.append(np.array(new_btag, dtype=np.double))
            # ### padding ###
            # add_num_of_events = max_events - len(new_event)
            # for i in range(add_num_of_events):
            #     new_event.append(np.array(
            #         [0 for i in range(len(particle_id_fields) + 2 + len(fields))], dtype=np.double))
            # ### end of padding ###
            new_vals.append(np.vstack(new_event))
        formatted_data[label] = new_vals
    return formatted_data
