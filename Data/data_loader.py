import torch.utils.data

from Data.data_reader import DataReader


class DataLoader(torch.utils.data.dataset):
    def __init__(self, typ, **kwargs):
        if typ not in ['train', 'test']:
            raise ValueError("Invalid typ: {} \n The typ has to be test or train".format(typ))
        self.aug = kwargs.get('augmentation')
        print("Initiating the {} data".format(typ))
        self.datareader = DataReader()

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        return self.datareader.read_data(str(item))
