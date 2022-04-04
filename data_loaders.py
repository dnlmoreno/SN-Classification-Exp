import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class PhotometryDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, one_hot):
        self.data = data

        if one_hot:
            self.labels = labels.float() # One-hot
        else:
            self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class PhotometryDataLoaders(torch.utils.data.DataLoader):
    def __init__(self, X_train, y_train, X_test, y_test, one_hot, batch_size=256,# X_val, y_val,
                num_workers=1, shuffle=False, collate_fn=None, normalize=False, n_quantiles=1000,
                weight_norm=False):

        self.X_train = X_train
        #self.X_val = X_val
        self.X_test = X_test

        self.y_train = y_train
        #self.y_val = y_val
        self.y_test = y_test

        self.one_hot = one_hot

        #if normalize == True:
        #    self.train_data, self.val_data, self.test_data = self.__normalizeData(n_quantiles)

        self.__loader(batch_size, num_workers, shuffle, collate_fn, weight_norm)

    def __loader(self, batch_size, num_workers, shuffle, collate_fn, weight_norm):
        train_set = PhotometryDataset(self.X_train, self.y_train, self.one_hot)
        #val_set = PhotometryDataset(self.X_val, self.y_val)
        test_set = PhotometryDataset(self.X_test, self.y_test, self.one_hot)

        sampler=None
        if weight_norm:
            weight_labels = self.__weights_for_balanced_classes(train_set)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weight_labels,
                                                                    num_samples=len(train_set),
                                                                    replacement=True)

        self.train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=num_workers, collate_fn=collate_fn, sampler=sampler)

        #self.val = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=shuffle, 
        #                                       num_workers=num_workers, collate_fn=collate_fn)

        self.test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=num_workers, collate_fn=collate_fn)

    def __weights_for_balanced_classes(self, train_set):
        weights = train_set.labels.bincount()
        weights = [1/i for i in weights]

        weight_labels = []
        for label in train_set.labels:
            for category in range(len(weights)):
                if label == category:
                    weight_labels.append(weights[category])
        
        return torch.Tensor(weight_labels)


import numpy as np

def collate_fn(data):
    list_lenght = torch.zeros(len(data))
    for i in range(len(data)):
        list_lenght[i] = len(data[i][0])

    idx_sorted = list_lenght.argsort(descending=True)

    dataset_sorted = []
    for idx in idx_sorted:
        dataset_sorted.append(data[idx])

    seq_len = list_lenght[idx_sorted]

    data_list, label_list = [], []
    for (_data, _label) in dataset_sorted:
        data_list.append(_data)
        label_list.append(_label)

    data_list = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=-999.0).float()
    data_list = torch.nn.utils.rnn.pack_padded_sequence(data_list, seq_len, batch_first=True)

    return data_list, label_list, idx_sorted


def collate_fn_exp_1(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    seq_len = [s[0].size(0) for s in data]

    data_list, label_list = [], []
    
    for (_data, _label) in data:
        data_list.append(_data)
        label_list.append(_label)

    data_list = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=-999.0).float()
    data_list = torch.nn.utils.rnn.pack_padded_sequence(data_list, seq_len, batch_first=True)

    return data_list, label_list 


class PhotometryUnlabeledDataLoaders(torch.utils.data.DataLoader):
    def __init__(self, X_unlab_train, teacher_model, batch_size=256,# X_val, y_val,
                num_workers=8, shuffle=False, collate_fn=None):
        self.teacher_model = teacher_model
        self.X_unlab_train = X_unlab_train

        self.__loader(batch_size, num_workers, shuffle, collate_fn)

    def create_soft_labels(self):
        pass


    def __loader(self, batch_size, num_workers, shuffle, collate_fn):
        train_set = PhotometryDataset(self.X_train, self.y_train)
        #val_set = PhotometryDataset(self.X_val, self.y_val)
        test_set = PhotometryDataset(self.X_test, self.y_test)

        self.train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=num_workers, collate_fn=collate_fn)

        #self.val = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=shuffle, 
        #                                       num_workers=num_workers, collate_fn=collate_fn)

        self.test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, 
                                                    num_workers=num_workers, collate_fn=collate_fn)