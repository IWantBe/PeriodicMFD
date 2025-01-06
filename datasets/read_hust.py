'''to read hust dataset and some preprocessing, there 4 load in hust and 9 fault status

4 loads are: 65Hz，70Hz，75Hz，80Hz

9 fault status are: normal, 
                    medium inner race fault, 
                    serious inner race fault, 
                    medium outter race fault, 
                    serious outter race fault, 
                    medium ball fault, 
                    serious ball fault, 
                    medium combo fault, 
                    serious combo fault

in dataset files, the files with 0.5X prefix are medium fault, otherwise are serious fault, 
                  the files with H prefix are normal, 
                  the files with I prefix are inner race fault, 
                  the files with O prefix are outter race fault, 
                  the files with B prefix are ball fault, 
                  the files with C prefix are combo fault'''

import torch
from torch.utils.data import TensorDataset

import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from pathlib import Path


def read_hust_data(
        load: int,  # load, 65, 70, 75, 80
):
    '''Read the data corresponding to the load in the HUST dataset'''

    HUST_path = Path(__file__).resolve().parent / 'data_hust'  # HUST dataset path

    filenames = [  # 9 fault category files under corresponding loads
        f'H_{load}Hz.xls',  # normal
        f'0.5X_I_{load}Hz.xls',  # medium inner race fault
        f'I_{load}Hz.xls',  # serious inner race fault
        f'0.5X_O_{load}Hz.xls',  # medium outter race fault
        f'O_{load}Hz.xls',  # serious outter race fault
        f'0.5X_B_{load}Hz.xls',  # medium ball fault
        f'B_{load}Hz.xls',  # serious ball fault
        f'0.5X_C_{load}Hz.xls',  # medium combo fault
        f'C_{load}Hz.xls',  # serious combo fault
    ]

    data = []

    for filename in filenames:
        filepath = str(HUST_path / filename)
        with open(filepath) as f:
            contents = f.readlines()
        contents = list(map(lambda x: x.strip(), contents))

        for i, content in enumerate(contents):  # find the data part
            if content == 'Data':
                contents = contents[i + 1:]
                break

        da = []
        for i, content in enumerate(contents):
            content = list(map(lambda x: float(x), content.split()[2:]))
            da.append(content)
        da = np.array(da, dtype=np.float64)  # L x 3

        data.append(da)

    # data  9 x L x 3
    data = list(map(lambda x: x.transpose(1, 0), data))  # 9 x 3 x L

    return data


def read_hust(
        load: int,  # load
        sample_len: int,  # sample length
        nspc: int,  # sample number of every class
        norm: str = None,  # 'first', 'after'  where to do normalization
):
    '''read hust dataset'''

    data = read_hust_data(load)  # 9 × 3 × L
    num_classes = len(data)  # 9

    if norm == 'first':
        for i in range(num_classes):
            data[i] = StandardScaler().fit_transform(data[i].T).T  # zero-mean normalization

    x, y = [], []
    for i in range(num_classes):
        ix, iy = [], []
        stride = (data[i].shape[1] - sample_len) // (nspc - 1)
        for j in range(nspc):
            ix.append(data[i][:, j * stride:j * stride + sample_len].tolist())
            iy.append(i)
        if len(ix) > nspc:
            ix = ix[:nspc]
            iy = iy[:nspc]
        x.extend(ix)
        y.extend(iy)

    if norm == 'after':
        x = np.array(x, dtype=np.float64)
        xl = []
        for i in range(x.shape[1]):
            xx = x[:, i, :]
            xx = StandardScaler().fit_transform(xx.T).T[:, np.newaxis, :]
            xl.append(xx)
        x = np.concatenate(xl, axis=1).tolist()

    return x, y  # N sample × 3 × sample_len, N


def read_hust_ap(
        load: int,  # load
        sample_len: int,  # sample length
        nspc: int,  # sample number of every class
        norm: str = None,  # 'first', 'after'  where to do normalization
        feature_scaling: bool = True,  # do or not feature scaling, not then do sample with fix length again
):
    '''sample and feature scaling based hust'''

    if load == 65: cycle = 394
    elif load == 70: cycle = 366
    elif load == 75: cycle = 342
    elif load == 80: cycle = 320

    # periodic sampling
    pn = math.ceil(cycle / sample_len)  # how much points to calculate mean
    sl = pn * sample_len  # sample length in sampling
    x, y = read_hust(load, sl, nspc, norm=norm)  # N sample × 3 × sl, N

    if feature_scaling:
        for i, v in enumerate(x):
            v = v[0]  # get first channel
            x[i] = [[sum(v[j * pn:(j + 1) * pn]) / pn for j in range(sample_len)]]
    else:
        # sampling with fixed length
        for i, v in enumerate(x):
            v = v[0]
            x[i] = [v[:sample_len]]

    return x, y  # N sample × 1 × sample_len, N


##########################################################################################


def load_dataset_2src_hust(
        src1: int,  # source domain 1
        src2: int,  # source domain 2
        target: int,  # target domain
        scaling: bool,  # feature scaling or not
        sample_len: int,  # sample length finally
        nspc: int,  # sample number of every class
        rate: list[float],  # split ratio in test set
        union: bool = False,  # add together 2 source as 1 train set or not
        feature_scaling: bool = True,  # do or not feature scaling, not then do sample with fix length again
):
    '''load hust - 2 source'''

    readfunc = read_hust_ap if scaling else read_hust

    if readfunc == read_hust_ap:
        x_s1, y_s1 = readfunc(src1, sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_s1, y_s1 = readfunc(src1, sample_len, nspc, norm='first')
    x_s1 = torch.tensor(x_s1, dtype=torch.float)[:, 0:1, :]
    y_s1 = torch.tensor(y_s1, dtype=torch.long)

    if readfunc == read_hust_ap:
        x_s2, y_s2 = readfunc(src2, sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_s2, y_s2 = readfunc(src2, sample_len, nspc, norm='first')
    x_s2 = torch.tensor(x_s2, dtype=torch.float)[:, 0:1, :]
    y_s2 = torch.tensor(y_s2, dtype=torch.long)

    if union:
        x_s = torch.cat([x_s1, x_s2], 0)
        y_s = torch.cat([y_s1, y_s2], 0)

    if readfunc == read_hust_ap:
        x_t, y_t = readfunc(target, sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_t, y_t = readfunc(target, sample_len, nspc, norm='first')
    x_t = torch.tensor(x_t, dtype=torch.float)[:, 0:1, :]
    y_t = torch.tensor(y_t, dtype=torch.long)

    x = [[] for _ in range(9)]
    y = [[] for _ in range(9)]
    for i, v in enumerate(x_t):
        x[y_t[i]].append(v)
    for i in range(len(y)):
        y[i] = [i for _ in range(len(x[i]))]
    x_t_train = []
    for i in range(len(x)):
        x_t_train.extend(x[i][:int(len(x[i]) * rate[0])])
    x_t_train = torch.cat(x_t_train).unsqueeze(1)
    y_t_train = []
    for i in range(len(y)):
        y_t_train.extend(y[i][:int(len(y[i]) * rate[0])])
    y_t_train = torch.tensor(y_t_train)
    x_t_test = []
    for i in range(len(x)):
        x_t_test.extend(x[i][int(len(x[i]) * rate[0]):])
    x_t_test = torch.cat(x_t_test).unsqueeze(1)
    y_t_test = []
    for i in range(len(y)):
        y_t_test.extend(y[i][int(len(y[i]) * rate[0]):])
    y_t_test = torch.tensor(y_t_test)

    if union:
        print(x_s.shape, y_s.shape, x_t_train.shape, y_t_train.shape, x_t_test.shape, y_t_test.shape)
    else:
        print(x_s1.shape, y_s1.shape, x_s2.shape, y_s2.shape, x_t_train.shape, y_t_train.shape, x_t_test.shape, y_t_test.shape)

    if union:
        dataset_source = TensorDataset(x_s, y_s)
        dataset_target_train = TensorDataset(x_t_train, y_t_train)
        dataset_target_test = TensorDataset(x_t_test, y_t_test)
        return dataset_source, dataset_target_train, dataset_target_test
    else:
        dataset_source1 = TensorDataset(x_s1, y_s1)
        dataset_source2 = TensorDataset(x_s2, y_s2)
        dataset_target_train = TensorDataset(x_t_train, y_t_train)
        dataset_target_test = TensorDataset(x_t_test, y_t_test)
        return dataset_source1, dataset_source2, dataset_target_train, dataset_target_test
