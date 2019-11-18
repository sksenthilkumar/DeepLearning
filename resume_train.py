import pathlib
import argparse
import pandas as pd

import utils
from train import Train


class ResumeTrain(Train):
    def __init__(self, exp_fol):
        self.exp_fol = pathlib.Path(exp_fol)
        if not self.exp_fol.exists():
            raise FileNotFoundError("The given experiment folder path does not exists: {}".format(str(self.exp_fol)))

        train_config = self.exp_fol.joinpath("train_config.yaml")
        Train.__init__(self, config_file=str(train_config))

    def setup_model(self):
        self.model = utils.load_model_frm_exp(str(self.exp_fol))

    def setup_opfol(self):
        # setting up the output folder
        self.op_path = self.exp_fol
        self.op_fol = self.op_path.name
        [self.exp_id, self.exp_name] = self.op_fol.split('_')

    def setup_logdf(self):
        logdf = pd.read_csv(str(self.op_path.joinpath("epoch_logs.csv")))
        # the last max e_acc will be saved. So all the entries in log df after the best epoch should be deleted
        best_epoch_idx = logdf['test_acc'].idxmax()
        self.logdf = logdf[:best_epoch_idx+1]
        self.start_epoch = best_epoch_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_fol", required=True, help="experiment folder")
    args = parser.parse_args()
    a = ResumeTrain(args.exp_fol)
    a.init_setup()
    a.start_training()