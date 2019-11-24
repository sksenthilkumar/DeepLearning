import pathlib
import random
import numpy as np
from PIL import Image

import scipy.io as scio
import torch.utils.data
import matplotlib.pyplot as plt


class Market1501(torch.utils.data.Dataset):
    """
    Market1501 is a child of class pytorch dataset is used to load the images and attribute dataset from
    Market 1501 dataset
    """
    def __init__(self, typ, **kwargs):

        if typ not in ['train', 'test']:
            raise ValueError("Invalid typ: {} \n The typ has to be test or train".format(typ))
        print("Initiating the Market 1501 {}ing data".format(typ))

        # constants
        path_dic = {'senthil': 'data/Market-1501-v15.09.15'}  # user name, and path to the dataset
        attributes_fol_name = 'market_attribute.mat'

        # get arguments
        self.aug = kwargs.get('augmentation')

        self.home_path = pathlib.Path.home()
        username = self.home_path.stem

        self.path = self.home_path.joinpath(path_dic[username])
        self.data_fol = self.path.joinpath('bounding_box_{}'.format(typ))
        self.data_list = list(self.data_fol.glob("*.jpg"))
        self.attributes = MrktAttribute(str(self.path.joinpath(attributes_fol_name)))
        self.get_atts_of = getattr(self.attributes, "get_{}_atts_of".format(typ))

    def show_sample(self):
        idx = random.randint(0, self.__len__())
        data, id = self.__getitem__(idx)
        data.show()
        atts = self.get_atts_of(id)
        for idx, val in enumerate(list(atts)):
            print("{:.^15}: {}".format(self.attributes.names[idx], val))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = MrktImage(self.data_list[idx])
        return data.image, data.id, self.get_atts_of(data.id)


class MrktImage:
    def __init__(self, path):
        if not isinstance(path, pathlib.Path):
            self.path = pathlib.Path(path)
        else:
            self.path = path

        self.name = self.path.stem
        self.id = self.name.split('_')[0]
        self.__image__ = None

    @property
    def image(self):
        if not self.__image__:
            with open(str(self.path), 'rb') as f:
                img = Image.open(f)
                self.__image__ = img.convert('RGB')

        return self.__image__


class MrktAttribute:
    def __init__(self, path):
        # constants
        self.names = ['age', 'backpack', 'bag', 'handbag', 'downblack', 'downblue', 'downbrown', 'downgray',
                      'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow', 'upblack', 'upblue',
                      'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow', 'clothes', 'down',
                      'up', 'hair', 'hat', 'gender', 'image_index']
        self.no_train_ids = 750
        self.no_test_ids = 751

        # variables
        self.path = path
        self.atts = scio.loadmat(str(self.path))['market_attribute'][0][0]

        self.train_atts= self.__get_atts__(typ='train')
        self.test_atts = self.__get_atts__(typ='test')

        self.train_ids = self.__get_ids__(typ='train')
        self.test_ids = self.__get_ids__(typ='test')

    def __get_ids__(self, typ):
        fo = getattr(self, '{}_atts'.format(typ))
        ids = fo[self.names.index('image_index'), :]
        return list(ids)

    def __get_atts__(self, typ):
        """
        Since the attribute values are in .mat format and loaded from python script, they do not have the expected
        shape of no_of_atts x no_of_data
        :param typ: test or train
        :return: returns an array of of shape no_of_atts x no_of_data
        """
        typ_atts = self.atts[0 if typ == 'test' else 1][0][0]

        new_atts = []
        for idx, i in enumerate(self.names):
            if i == 'image_index':
                fo = [i[0] for i in typ_atts[idx][0]]
                new_atts.append(fo)
            else:
                new_atts.append(list(typ_atts[idx])[0])

        return np.array(new_atts)

    def get_train_atts_of(self, id):
        return self.train_atts[:, self.train_ids.index(id)]

    def get_test_atts_of(self, id):
        return self.test_atts[:, self.test_ids.index(id)]

    def get_atts_of(self, id):
        """
        :param id: eg: between '0000' '1500'
        :return: the 28 attributes of the given id
        """
        if id in self.test_ids:
            return self.get_test_atts_of(id)
        elif id in self.train_ids:
            return self.get_train_atts_of(id)
        else:
            raise ValueError("Missing ID, the give id {} is not available".format(id))



