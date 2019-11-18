import utils
from Data.data_reader import  DataReader


class Infer():
    def __init__(self, exp_fol):

        self.model = utils.load_model_frm_exp(exp_fol)
        self.dr = DataReader()

    def infer(self, item):
        data = self.dr.read_data(item)
        out = self.model(data)
        return out