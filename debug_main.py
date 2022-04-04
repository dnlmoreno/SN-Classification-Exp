import random
import os

import torch
import numpy as np
import pandas as pd

# Lectura de los datos aumentados
path_data = os.path.abspath('data/spcc/supervised/charnock_preprocessed/unblind_hostz.csv')
num_augments = 5

# 0: supernovas tipo Ia
# 1: supernovas tipo NonIa
sn1a_classifier = {1:0, 2:1, 3:1, 21:1, 22: 1, 23:1, 32: 1, 33:1}


spcc_charnock_lc = pd.read_csv(path_data, header=None, sep=",")
spcc_charnock_lc.columns = ['oid', 'mjd', 'ra', 'decl', 'mwebv', 'photo_z', 'g_flux', 'r_flux', 'i_flux', 'z_flux', 'g_error', 'r_error', 'i_error', 'z_error', 'sim_redshift', 'class']

print(f"Cantidad de observaciones en un unblind_nohostz_1: {len(spcc_charnock_lc)}")
print(f"Cantidad de supernovas: {len(set(spcc_charnock_lc.oid))}")

spcc_charnock_lc['class'] = spcc_charnock_lc['class'].replace(sn1a_classifier)
print(f"Classes: {set(spcc_charnock_lc['class'])}")

# Cantidad y las supernovas que quedaran para el entrenamiento y testeo
test_fraction = 0.5

ids_sne = spcc_charnock_lc.oid.unique()
ids_sne_length = len(ids_sne)
test_length = int(ids_sne_length * test_fraction)
indices = np.random.permutation(ids_sne_length) # Indices permutados de los datos de ids_sne_length

# Indice de los datos que se entrenaran y testearan
training_idx, test_idx = indices[:ids_sne_length-test_length], indices[ids_sne_length-test_length:]

# ids de supernovas para entrenar y testear
ids_train = ids_sne[training_idx]
ids_test = ids_sne[test_idx]

print(f'Cantidad de supernovas en el conjunto de entrenamiento: {len(ids_train)}')
print(f'Cantidad de supernovas en el conjunto de prueba: {len(ids_test)}')

# Separa los datos
data, labels = [], []
training_idx = []
test_idx = []

idx = 0
for id in ids_sne:
    filter_sn = spcc_charnock_lc[spcc_charnock_lc.oid == id]
    data_sequence = filter_sn.iloc[0:, 1:-1:]

    # Genera copias exactas de la data original (aumentaciones de datos realizados en charnock)
    for augment in range(0, num_augments):
        labels.append(filter_sn.iloc[0,-1])
        data.append(torch.tensor(data_sequence.values))

        if id in ids_train:
            training_idx.append(idx)
        elif id in ids_test:
            test_idx.append(idx)

        idx += 1

# Desordena la consecutividad de las supernovas aumentadas
random.shuffle(training_idx)
random.shuffle(test_idx)


##################### Labels #####################
# In tensor format
labels_torch = torch.tensor(labels)
nb_classes = labels_torch.unique().size(0)

##################### Train data #####################
X_train = []
for idx in training_idx:
    X_train.append(data[idx])
    
#y_train = labels_torch[training_idx]
y_train = torch.nn.functional.one_hot(labels_torch[training_idx], nb_classes)

##################### Test data #####################
X_test = []
for idx in test_idx:
    X_test.append(data[idx])

#y_test = labels_torch[test_idx]
y_test = torch.nn.functional.one_hot(labels_torch[test_idx], nb_classes)

# one_hot: [SNIa, NonSNIa]
print(f'Cantidad de SNs: {len(data)}')
print(f'Cantidad de supernovas para training: {len(X_train)}')
print(f'Cantidad de supernovas para test: {len(X_test)}')


import data_loaders

data_loader = data_loaders.PhotometryDataLoaders(X_train, y_train, X_test, y_test, batch_size=10, 
                                                num_workers=0, shuffle=False, collate_fn=data_loaders.collate_fn)

# features and number classes in the dataset
input_size = data_loader.train_set.dataset.data[0].size(1)
num_classes = data_loader.train_set.dataset.labels.unique().size(0)

import conf

CONFIG = conf.get_args(input_size, num_classes)
CONFIG

import models.base_model

def model_type(CONFIG):
    
    if CONFIG.load_model:
        model = models.base_model.Model(CONFIG)
        model.summary()

    else:
        model = models.base_model.Model(CONFIG)
        model.summary()
        model.compile(CONFIG.loss_list, CONFIG.optimizer, CONFIG.lr, CONFIG.metric_eval)

    return model

charnock_model = model_type(CONFIG)

charnock_model.fit(data_loader.train_set, data_loader.test_set, 'normal', CONFIG.num_epochs, CONFIG.verbose)