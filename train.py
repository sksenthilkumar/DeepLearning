import yaml
import torch
import pathlib
import argparse
import datetime
import pandas as pd
from tqdm import tqdm
import torch.utils.data
from torch.autograd import Variable

from Data.data_loader import DataLoader
from utils import RunningAverage, top1_acc
from Model.model_factory import ModelFactory

torch.manual_seed(42)


# noinspection PyCallingNonCallable
class Train:
    def __init__(self, config_file):
        """
        Used to train pytorch models
        """
        self.train_config = yaml.full_load(open(config_file))
        self.train_params = self.train_config.get('train_params')
        self.model_params = self.train_config.get('model_params')

        # training setup
        self.total_epochs = self.train_params['epochs']
        self.batch_size = self.train_params['batch_size']
        self.learning_rate = self.train_params['lr']
        self.start_epoch = 0

    def init_setup(self):
        # initiate setups
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.setup_loss()
        self.setup_metric()
        self.setup_opfol()
        self.setup_logdf()

    def setup_model(self):
        self.model = ModelFactory(**self.model_params)

    def setup_data(self):
        # setting up the data
        self.train_data = torch.utils.data.DataLoader()
        self.test_data = torch.utils.data.DataLoader()

    def setup_optimizer(self):
        optimizer_name = self.train_params['optimizer']
        self.optimizer = None
        if optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        elif optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        else:
            raise ValueError("Unknown value {} for optimizer name".format(optimizer_name))

    def setup_loss(self):
        loss_function = self.train_params['loss']
        self.loss = None
        if loss_function == 'CE':
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("Unknown value {} for loss function".format(loss_function))

    def setup_metric(self):
        metric = self.train_params['metric']
        self.metric = None
        if metric == 'top1_acc':
            self.metric = top1_acc
        else:
            raise ValueError("Unknown value {} for metric".format(metric))

    def setup_opfol(self):
        # setting up the output folder
        self.exp_name = self.train_config.get('exp_name')
        self.exp_repo = self.train_config.get('save_path')
        self.exp_id = datetime.datetime.now().strftime("%y%m%d%H%M%S")
        self.op_fol = '_'.join([self.exp_id, self.exp_name])
        self.exp_repo = pathlib.Path(self.exp_repo)
        self.op_path = self.exp_repo.joinpath(self.op_fol)
        if not self.op_path.exists():
            self.op_path.mkdir(parents=True)

        # saving the training configuration file in the experiment folder
        save_config_path = self.op_path.joinpath("train_config.yaml")
        with open(str(save_config_path), 'w') as f:
            yaml.dump(self.train_config, f)

    def setup_logdf(self):
        self.logdf = pd.Dataframe(columns=['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
        self.save_log_df()

    def save_log_df(self):
        self.logdf.to_csv(str(self.op_path.joinpath("epoch_logs.csv")))

    def start_training(self):
        for i in range(self.start_epoch, self.total_epochs):
            acc, loss = self.one_epoch(mode='train', epoch_num=i)
            e_acc, e_loss = self.one_epoch(mode='test', epoch_num=i)

            self.logdf.append({'epoch': i, 'train_loss': loss(), 'test_loss': e_loss(),
                               'train_acc': acc(), 'test_acc': e_acc()})

            self.save_log_df()

            if e_acc() == self.logdf['eval_acc'].max():
                print("+++++++++SAVING+++++++++")
                model_save_path = self.op_path.joinpath("best.pth.tar")
                print("path : {}".format(model_save_path))
                torch.save({'epoch': i, 'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(), 'test_acc': e_acc()},
                           str(model_save_path))

    def one_epoch(self, mode, epoch_num):
        if mode not in ['train', 'test']:
            raise ValueError("Unknown value {} for mode".format(mode))
        print("{}ing... epoch: {}" .format(mode, epoch_num))

        if mode == 'train':
            self.model.train()
            dl = self.train_data
            one_iter_function = self.one_train_iteration
        else:
            self.model.eval()
            dl = self.test_data
            one_iter_function = self.one_test_iteration

        acc_avg = RunningAverage()
        loss_avg = RunningAverage()
        with tqdm(total=len(dl)) as t:
            for n, (data, label) in enumerate(dl):
                data, label = Variable(data), Variable(label)
                loss, acc = one_iter_function(data, label)
                loss_avg.update(loss)
                acc_avg.update(acc)
                t.set_postfix(run_param="Epoch{} Loss:{} Acc:{}".format(epoch_num, loss_avg(), acc_avg()))
                t.update()

        return acc_avg, loss_avg

    def one_train_iteration(self, data, label):
        self.optimizer.zero_grad()
        op = self.model(data)
        loss = self.loss(op, label)
        acc = self.metric(op, label)
        loss.backward()
        self.optimizer.step()

        return loss.data.item(), acc.item()

    def one_test_iteration(self, data, label):

        op = self.model(data)
        loss = self.loss(op, label)
        acc = self.metric(op, label)

        return loss.data.item(), acc.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tc", required=True, help="training configuration file")
    args = parser.parse_args()
    a = Train(args.tc)
    a.init_setup()
    a.start_training()
