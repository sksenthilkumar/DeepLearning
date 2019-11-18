import pathlib
import numpy as np


class DataReader:
    def __init__(self, src_path=""):
        self.src_path = src_path

    def read_data(self, name):
        return np.random.rand(3, 256, 256)