'''
PeriodicMFD
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utilities as ut
import math
import model as M

from utilities import run_EX
from datasets import load_dataset_2src_cwru48k, load_dataset_2src_jnu, load_dataset_2src_hust
import argparse


class PeriodicMFD:

    def __init__(self, args: argparse.Namespace):
        self.w1, self.w2 = 0.5, 0.5
        self.min_loss = None
        self.args = args

    def __call__(self, src1: int, src2: int, target: int):

        ut.seed_everything(42)

        # hyper-parameters
        EPOCHS = self.args.epochs
        BATCH_SIZE = 32
        LR = 1e-3
        align = True
        nspc = 500

        if self.args.dataset == 'cwru48k':
            sample_len = 420 if self.args.sample_len <= 0 else self.args.sample_len
            num_classes = 10
        elif self.args.dataset == 'jnu':
            sample_len = 500 if self.args.sample_len <= 0 else self.args.sample_len
            num_classes = 4
        elif self.args.dataset == 'hust':
            sample_len = 400 if self.args.sample_len <= 0 else self.args.sample_len
            num_classes = 9

        rate = [0.9, 0.1]

        model = M.PeriodicMFDModel(sample_len, num_classes)
        model.w = nn.Parameter(torch.zeros(2))
        print(f'model params: {ut.get_params_num(model)}')  # model parameter quantity

        if self.args.dataset == 'cwru48k':  # dataset
            load_dataset_2src = load_dataset_2src_cwru48k
        elif self.args.dataset == 'jnu':
            load_dataset_2src = load_dataset_2src_jnu
        elif self.args.dataset == 'hust':
            load_dataset_2src = load_dataset_2src_hust

        # read data
        dataset_s1, dataset_s2, dataset_t_train, dataset_t_test = load_dataset_2src(src1, src2, target, align, sample_len, nspc, rate)
        dataloader_s1 = DataLoader(dataset_s1, BATCH_SIZE, True, drop_last=True)
        dataloader_s2 = DataLoader(dataset_s2, BATCH_SIZE, True, drop_last=True)
        dataloader_t_train = DataLoader(dataset_t_train, BATCH_SIZE, True, drop_last=True)
        dataloader_t_test = DataLoader(dataset_t_test, BATCH_SIZE, True, drop_last=True)

        # train device
        device = self.args.device
        print(f'Using {device}')

        # train model
        model = model.to(device)
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        for epoch in range(1, EPOCHS + 1):

            dataloader_s_it1 = iter(dataloader_s1)
            dataloader_s_it2 = iter(dataloader_s2)
            dataloader_t_it = iter(dataloader_t_train)
            len_s1 = len(dataloader_s1)
            len_s2 = len(dataloader_s2)
            len_t = len(dataloader_t_train)
            length = max(len_s1, len_s2, len_t)

            mean_loss_c1 = 0
            mean_loss_c2 = 0
            mean_loss_L1_1 = 0
            mean_loss_L1_2 = 0
            mean_loss_mcc1 = 0
            mean_loss_mcc2 = 0
            model.train()
            for i in range(length):
                try:
                    x_s1, y_s1 = next(dataloader_s_it1)
                except StopIteration:
                    dataloader_s_it1 = iter(dataloader_s1)
                    x_s1, y_s1 = next(dataloader_s_it1)
                try:
                    x_s2, y_s2 = next(dataloader_s_it2)
                except StopIteration:
                    dataloader_s_it2 = iter(dataloader_s2)
                    x_s2, y_s2 = next(dataloader_s_it2)
                try:
                    x_t, y_t = next(dataloader_t_it)
                except StopIteration:
                    dataloader_t_it = iter(dataloader_t_train)
                    x_t, y_t = next(dataloader_t_it)
                x_s1 = x_s1.to(device)
                y_s1 = y_s1.to(device)
                x_s2 = x_s2.to(device)
                y_s2 = y_s2.to(device)
                x_t = x_t.to(device)
                y_t = y_t.to(device)

                gamma = 2 / (1 + math.exp(-10 * (epoch) / (EPOCHS))) - 1

                ce1, J_dis1, MCC1 = model(x_s1, x_t, y_s1, mark=1)
                ce2, J_dis2, MCC2 = model(x_s2, x_t, y_s2, mark=2)

                w1, w2 = model.w.softmax(0)

                J_dis = J_dis1 + J_dis2
                J_mcu = w1 * MCC1 + w2 * MCC2
                J_ce = w1 * ce1 + w2 * ce2

                loss = J_ce + gamma * (J_mcu + J_dis)

                if self.min_loss == None or loss.item() < self.min_loss:
                    self.min_loss = loss.item()
                    self.w1, self.w2 = model.w[0].item(), model.w[1].item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mean_loss_c1 += ce1.item()
                mean_loss_c2 += ce2.item()
                mean_loss_L1_1 += J_dis1.item()
                mean_loss_L1_2 += J_dis2.item()
                mean_loss_mcc1 += MCC1.item()
                mean_loss_mcc2 += MCC2.item()
            mean_loss_c1 /= length
            mean_loss_c2 /= length
            mean_loss_L1_1 /= length
            mean_loss_L1_2 /= length
            mean_loss_mcc1 /= length
            mean_loss_mcc2 /= length

            w1, w2 = torch.tensor([self.w1, self.w2]).softmax(0)

            test_loss1 = 0
            test_loss2 = 0
            test_loss = 0
            test_acc1 = 0
            test_acc2 = 0
            test_acc = 0
            model.eval()
            len_t = len(dataloader_t_test)
            for i, (x, y) in enumerate(dataloader_t_test):
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    out1, out2 = model(x)

                    out = w1 * out1 + w2 * out2

                    loss1 = ce_loss(out1, y)
                    loss2 = ce_loss(out2, y)
                    loss = ce_loss(out, y)
                test_loss1 += loss1.item()
                test_loss2 += loss2.item()
                test_loss += loss.item()
                test_acc1 += (out1.argmax(1) == y).sum().item()
                test_acc2 += (out2.argmax(1) == y).sum().item()
                test_acc += (out.argmax(1) == y).sum().item()
            test_loss1 /= len_t
            test_loss2 /= len_t
            test_loss /= len_t
            test_acc1 /= len_t * BATCH_SIZE
            test_acc2 /= len_t * BATCH_SIZE
            test_acc /= len_t * BATCH_SIZE

            print(f'Epoch: {epoch}/{EPOCHS}, '
                  f'loss c1: {mean_loss_c1:.5f}, '
                  f'loss c2: {mean_loss_c2:.5f}, '
                  f'loss L1 1: {mean_loss_L1_1:.5f}, '
                  f'loss L1 2: {mean_loss_L1_2:.5f}, '
                  f'loss mcc1: {mean_loss_mcc1:.5f}, '
                  f'loss mcc2: {mean_loss_mcc2:.5f}, '
                  f'w1: {w1.item():.5f}, '
                  f'w2: {w2.item():.5f}, '
                  f'loss c test1: {test_loss1:.5f}, '
                  f'loss c test2: {test_loss2:.5f}, '
                  f'loss c test: {test_loss:.5f}, '
                  f'test acc1: {test_acc1:.5f}, test acc2: {test_acc2:.5f}, test acc: {test_acc:.5f}\n')

        model.w = nn.Parameter(torch.tensor([self.w1, self.w2]))

        if self.args.save_weights:
            torch.save(model.state_dict(), self.save_weights)

        return test_acc


if __name__ == '__main__':

    # run model in cwru48k
    # python PeriodicMFD.py --dataset cwru48k --tasks [0,1,2] [0,1,3] [0,2,1] [0,2,3] [0,3,1] [0,3,2] [1,2,0] [1,2,3] [1,3,0] [1,3,2] [2,3,0] [2,3,1]

    # run model in jnu
    # python PeriodicMFD.py --dataset jnu --tasks [600,800,1000] [600,1000,800] [800,1000,600]

    # run model in hust
    # python PeriodicMFD.py --dataset hust --tasks [65,70,75] [65,70,80] [65,75,70] [65,75,80] [65,80,70] [65,80,75] [70,75,65] [70,75,80] [70,80,65] [70,80,75] [75,80,65] [75,80,70]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset: cwru48k, jnu, hust')
    parser.add_argument('--save_weights', type=str, default='', help='save model weights path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--sample_len', type=int, default=0, help='if sample_len <= 0, auto chose sample len of every dataset')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--run', type=int, default=10, help='how much times run')
    parser.add_argument('--tasks', type=str, nargs='+')
    args = parser.parse_args()

    for i, task in enumerate(args.tasks):
        args.tasks[i] = eval(task)

    run_EX(args.tasks, [PeriodicMFD(args)], args.run)
