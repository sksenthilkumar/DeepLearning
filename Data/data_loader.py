import pathlib
import random
import numpy as np
from PIL import Image

import scipy.io as scio
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


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
        path_dic = {'senthil': 'data/Market-1501-v15.09.15',  # user name, and path to the dataset
                    'skathiresan': '/media/skathiresan/Data/Market-1501-v15.09.15'
                    }
        attributes_fol_name = 'market_attribute.mat'

        # get arguments
        self.aug = kwargs.get('augmentation')

        self.home_path = pathlib.Path.home()
        username = self.home_path.stem

        self.path = self.home_path.joinpath(path_dic[username])
        self.data_fol = self.path.joinpath('bounding_box_{}'.format(typ))

        self.attributes = MrktAttribute(str(self.path.joinpath(attributes_fol_name)))
        self.get_atts_of = getattr(self.attributes, "get_{}_atts_of".format(typ))

        self.data_list = self.get_data_list()

        # transformation list
        self.train_transformation = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
        ])

        self.test_transformation = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.transformation = getattr(self, '{}_transformation'.format(typ))

    def get_data_list(self):
        data_list = []
        fo = list(self.data_fol.glob("*.jpg"))
        images_not_in_list = 0
        for i in fo:
            if not int(i.stem.split('_')[0]) in [0, -1]:
                data_list.append(i)
            else:
                images_not_in_list += 1

        print("{} images with class -1/0 are not added to data list".format(images_not_in_list))

        return data_list

    def show_sample(self):
        idx = random.randint(0, self.__len__())
        data, _id = self.__getitem__(idx)
        data.show()
        atts = self.get_atts_of(_id)
        for idx, val in enumerate(list(atts)):
            print("{:.^15}: {}".format(self.attributes.names[idx], val))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = MrktImage(self.data_list[idx])
        return data.numpy(for_torch_training=True), self.get_atts_of(data.id)


class MrktImage:
    def __init__(self, path):
        if not isinstance(path, pathlib.Path):
            self.path = pathlib.Path(path)
        else:
            self.path = path

        self.name = self.path.stem
        self.id = int(self.name.split('_')[0])
        self.__image__ = None

    @property
    def image(self):
        if not self.__image__:
            with open(str(self.path), 'rb') as f:
                img = Image.open(f)
                self.__image__ = img.convert('RGB')

        return self.__image__

    def numpy(self, for_torch_training=False):
        img = np.array(self.image)
        if for_torch_training:
            img = np.transpose(img, (2, 0, 1))
        return img


class MrktAttribute:
    def __init__(self, path=None):
        # constants
        self.names = ['age', 'backpack', 'bag', 'handbag', 'downblack', 'downblue', 'downbrown', 'downgray',
                      'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow', 'upblack', 'upblue',
                      'upgreen', 'upgray', 'uppurple', 'upred', 'upwhite', 'upyellow', 'clothes', 'down',
                      'up', 'hair', 'hat', 'gender', 'image_index']

        # actual number of atts with number of values representing the atts
        self.actual_atts = {'age': 1, 'backpack': 1, 'bag': 1, 'handbag': 1, 'down_color': 9, 'up_color': 8,
                            'clothes': 1, 'down': 1, 'up': 1, 'hair': 1, 'hat': 1, 'gender': 1}

        # actual number of atts with number of required output neurons in the ne (important case: 'age')
        self.atts_op_neurons = {'age': 4, 'backpack': 1, 'bag': 1, 'handbag': 1, 'down_color': 9, 'up_color': 8,
                                'clothes': 1, 'down': 1, 'up': 1, 'hair': 1, 'hat': 1, 'gender': 1}
        self.total_op_neurons = sum(list(self.atts_op_neurons.values()))

        # attributes label type: 'i' = integer, 'b' = binary, 'o' = one_hot
        self.atts_label_type = {'age': 'i', 'backpack': 'b', 'bag': 'b', 'handbag': 'b', 'down_color': 'o',
                                'up_color': 'o', 'clothes': 'b', 'down': 'b', 'up': 'b', 'hair': 'b', 'hat': 'b',
                                'gender': 'b'}

        self.no_train_ids = 750
        self.no_test_ids = 751

        # variables
        if path:
            self.path = path
            self.atts = scio.loadmat(str(self.path))['market_attribute'][0][0]

            self.train_atts = self.__get_atts__(typ='train')
            self.test_atts = self.__get_atts__(typ='test')

            self.train_ids = self.__get_ids__(typ='train')
            self.test_ids = self.__get_ids__(typ='test')
        else:
            print("No path value")

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

        atts = np.array(new_atts, dtype=int)
        temp = atts[:-1, :] - 1
        atts[:-1, :] = temp
        return atts

    def get_train_atts_of(self, _id, leave_index=False):
        ans = self.train_atts[:, self.train_ids.index(_id)]
        if leave_index:
            ans = ans[:-1]
        return ans

    def get_test_atts_of(self, _id, leave_index=False):
        ans = self.test_atts[:, self.test_ids.index(_id)]
        if leave_index:
            ans = ans[:-1]
        return ans

    def get_atts_of(self, _id):
        """
        :param _id: eg: between '0000' '1500'
        :return: the 28 attributes of the given id
        """
        if _id in self.test_ids:
            return self.get_test_atts_of(_id)
        elif _id in self.train_ids:
            return self.get_train_atts_of(_id)
        else:
            raise ValueError("Missing ID, the give id {} is not available".format(_id))
