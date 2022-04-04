""" 
Script que divide los ids de las curvas de luz dentro de un conjunto de entrenamiento, 
validación y test.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold

class LightCurvesSplit(object):
    def __init__(self, name_col_id, name_col_label):
        self.name_col_id = name_col_id
        self.name_col_label = name_col_label

    def id_and_label_ligthcurves(self, df_lc):
        """Crea un dataframe con el id de las supernovas y su respectivo label"""
        list_ids = df_lc[self.name_col_id].unique()
        id_sn = []
        labels_sn = []
        for id in list_ids:
            filter_lc = df_lc[df_lc[self.name_col_id] == id]
            id_sn.append(id)
            labels_sn.append(filter_lc[self.name_col_label].iloc[0])

        df = pd.DataFrame(zip(id_sn, labels_sn), columns=[self.name_col_id, self.name_col_label])
        self.df_dataset = df.assign(fold='')

        return df

    def test_split(self, test_size=200): # Esto es create label, tengo que crear los datos de test en las columnas
        """Implementado solo para dataset con labels binarios"""
        type = 'test'
        self.test_set = self.__data_split(test_size, type)
        
        return self.test_set

    def create_unlabeled_data(self, data_size=400):
        type = 'unlabeled'
        df_data_created = self.__data_split(data_size, type)

        return df_data_created.drop(['fold'], axis=1)

    def train_val_split_stratified(self, n_splits=5, random_state=1): #, train_size=0.8
        # sklearn.model_selection.StratifiedKFold
        split_shuffles = StratifiedKFold(n_splits=n_splits,
                                        #test_size=1-train_size,
                                        #train_size=train_size,
                                        shuffle = True,
                                        random_state=random_state)

        aux_df_dataset = self.df_dataset[self.df_dataset['fold'] != 'test']
        split_stratified = list(split_shuffles.split(aux_df_dataset[self.name_col_id], 
                                                    aux_df_dataset[self.name_col_label]))

        k_dataset_list = []
        for n_split in range(n_splits):
            k_dataset_list.append(self.df_dataset.copy())
            train_idx, test_idx = split_stratified[n_split]

            k_dataset_list[n_split]['fold'].iloc[train_idx] = f'training_{n_split}'
            k_dataset_list[n_split]['fold'].iloc[test_idx] = f'validation_{n_split}'

        k_fold_objects = pd.concat([pd.concat(k_dataset_list), self.test_set])

        return k_fold_objects

    def __data_split(self, size, type):
        list_classes = self.df_dataset[self.name_col_label].unique()
        num_classes = len(list_classes)
        sample_per_class = int(size/num_classes)

        aux_list = []
        for clase in list_classes:
            filter_type = self.df_dataset[self.df_dataset[self.name_col_label] == clase]
            df_sample = filter_type.sample(sample_per_class, random_state=1)
            self.df_dataset = self.df_dataset.drop(df_sample.index)
            aux_list.append(df_sample)

        self.df_dataset = self.df_dataset.reset_index(drop=True)
        df_data_created = pd.concat(aux_list).reset_index(drop=True)

        if type == 'test': df_data_created.fold = 'test'
        return df_data_created