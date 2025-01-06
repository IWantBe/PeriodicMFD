import torch
from torch.utils.data import TensorDataset

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import hdf5storage
from pathlib import Path


def read_jnu_s(
        path: str,  # dataset folder path
        sample_len: int,  # sample length
        nspc: int = 1000,  # sample number of every class
        norm: str = None,  # 'first', 'after'  where to do normalization
):
    data = {0: [], 1: [], 2: [], 3: []}

    filenames = os.listdir(path)  # get fault data file name of all fault class
    for filename in filenames:  # traverse data of all files
        filepath = os.path.join(path, filename)
        with open(filepath) as f:
            content = f.readlines()
        content = list(map(lambda v: float(v.strip()), content))  # del all space char and convert it to float

        # label setting
        if filename[0] == 'n': k = 0  # [1,0,0,0] normal
        elif filename[0] == 'i': k = 1  # [0,1,0,0] innerrace fault
        elif filename[0] == 'o': k = 2  # [0,0,1,0] outerrace fault
        elif filename[0] == 't': k = 3  # [0,0,0,1] ball fault
        data[k].append(content)

    for i in range(4):  # convert to ndarray
        data[i] = np.array(data[i], dtype=np.float64)

    if norm == 'first':
        for i in range(4):
            data[i] = StandardScaler().fit_transform(data[i].T).T  # zero-mean normalization

    x, y = [], []
    for i in range(4):
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
        x = np.array(x, dtype=np.float64).squeeze(1)
        x = StandardScaler().fit_transform(x.T).T
        x = x[:, np.newaxis, :].tolist()

    return x, y  # N×1×L, N


def read_jnu_ap(
        path: str,
        sample_len: int,  # sample length
        nspc: int,  # sample number of every class
        norm: str = None,  # 'first', 'after'  where to do normalization
        feature_scaling: bool = True,  # do or not feature scaling, not then do sample with fix length again
):
    '''sample and feature scaling'''

    path = os.path.normpath(path)
    load = int(os.path.split(path)[1])
    if load == 600: cycle = 5000
    elif load == 800: cycle = 3750
    elif load == 1000: cycle = 3000

    # periodic sampling
    pn = math.ceil(cycle / sample_len)  # how much points to calculate mean
    sl = pn * sample_len  # sample length in sampling
    x, y = read_jnu_s(path, sl, nspc, norm=norm)  # N × 1 × sl, N

    if feature_scaling:
        for i, v in enumerate(x):
            v = v[0]
            x[i] = [[sum(v[j * pn:(j + 1) * pn]) / pn for j in range(sample_len)]]
    else:
        # sampling with fixed length
        for i, v in enumerate(x):
            v = v[0]
            x[i] = [v[:sample_len]]

    return x, y  # N × 1 × sample_len, N


def read_cwru_data(path: str):

    data = [[] for _ in range(10)]

    filenames = os.listdir(path)
    for filename in filenames:  # traverse data of all files
        filepath = os.path.join(path, filename)
        mat = hdf5storage.loadmat(filepath)
        content = [[], []]
        for key in mat:
            if 'DE' in key:  # driver end
                content[0] = mat[key].ravel().tolist()
            elif 'FE' in key:  # fan end
                content[1] = mat[key].ravel().tolist()

        # label setting
        if 'normal' in filename: k = 0  # normal
        elif 'IR007' in filename: k = 1  # innerrace fault 007
        elif 'IR014' in filename: k = 2  # innerrace fault 014
        elif 'IR021' in filename: k = 3  # innerrace fault 021
        elif 'OR007' in filename: k = 4  # outerrace fault 007
        elif 'OR014' in filename: k = 5  # outerrace fault 014
        elif 'OR021' in filename: k = 6  # outerrace fault 021
        elif 'B007' in filename: k = 7  # ball fault 007
        elif 'B014' in filename: k = 8  # ball fault 014
        elif 'B021' in filename: k = 9  # ball fault 021
        data[k] = content

    # the first channel is DE, the second channel is FE
    return data  # 10 × 2 × L


def read_cwru_s10(
        path: str,  # dataset path
        sample_len: int,  # sample length
        nspc: int,  # sample number of every class
        norm: str = None,  # 'first', 'after'  where to do normalization
):
    '''read cwru dataset with 10 class'''

    data = read_cwru_data(path)  # 10 × 2 × L
    num_classes = len(data)  # 10

    for i in range(num_classes):  # convert to ndarray
        data[i] = np.array(data[i], dtype=np.float64)

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
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        x1 = StandardScaler().fit_transform(x1.T).T[:, np.newaxis, :]
        x2 = StandardScaler().fit_transform(x2.T).T[:, np.newaxis, :]
        x = np.concatenate([x1, x2], axis=1).tolist()

    return x, y  # N × 2 × L, N


def read_cwru_s10ap(
        path: str,
        sample_len: int,  # sample length
        nspc: int,  # sample number of every class
        norm: str = None,  # 'first', 'after'  where to do normalization
        feature_scaling: bool = True,  # do or not feature scaling, not then do sample with fix length again
):
    '''sample and feature scaling based cwru with 10 class'''

    path = os.path.normpath(path)
    load = int(path.split(os.sep)[-1][0])
    if load == 0: cycle = 1603
    elif load == 1: cycle = 1625
    elif load == 2: cycle = 1646
    elif load == 3: cycle = 1665

    # periodic sampling
    pn = math.ceil(cycle / sample_len)  # how much points to calculate mean
    sl = pn * sample_len  # sample length in sampling
    x, y = read_cwru_s10(path, sl, nspc, norm=norm)  # N × 2 × sl, N

    if feature_scaling:
        for i, v in enumerate(x):
            v = v[0]
            x[i] = [[sum(v[j * pn:(j + 1) * pn]) / pn for j in range(sample_len)]]
    else:
        # sampling with fixed length
        for i, v in enumerate(x):
            v = v[0]
            x[i] = [v[:sample_len]]

    return x, y  # N × 1 × sample_len, N


##########################################################################################


def load_dataset_2src_cwru48k(
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
    '''load cwru48k - 2 source'''

    readfunc = read_cwru_s10ap if scaling else read_cwru_s10

    path = Path(__file__).resolve().parent / 'data_cwru48k'

    if readfunc == read_cwru_s10ap:
        x_s1, y_s1 = readfunc(str(path / f'{src1}HP/'), sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_s1, y_s1 = readfunc(str(path / f'{src1}HP/'), sample_len, nspc, norm='first')
    x_s1 = torch.tensor(x_s1, dtype=torch.float)[:, 0:1, :]
    y_s1 = torch.tensor(y_s1, dtype=torch.long)

    if readfunc == read_cwru_s10ap:
        x_s2, y_s2 = readfunc(str(path / f'{src2}HP/'), sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_s2, y_s2 = readfunc(str(path / f'{src2}HP/'), sample_len, nspc, norm='first')
    x_s2 = torch.tensor(x_s2, dtype=torch.float)[:, 0:1, :]
    y_s2 = torch.tensor(y_s2, dtype=torch.long)

    if union:
        x_s = torch.cat([x_s1, x_s2], 0)
        y_s = torch.cat([y_s1, y_s2], 0)

    if readfunc == read_cwru_s10ap:
        x_t, y_t = readfunc(str(path / f'{target}HP/'), sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_t, y_t = readfunc(str(path / f'{target}HP/'), sample_len, nspc, norm='first')
    x_t = torch.tensor(x_t, dtype=torch.float)[:, 0:1, :]
    y_t = torch.tensor(y_t, dtype=torch.long)

    x = [[] for _ in range(10)]
    y = [[] for _ in range(10)]
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


def load_dataset_2src_jnu(
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
    '''load jnu - 2 source'''

    readfunc = read_jnu_ap if scaling else read_jnu_s

    path = Path(__file__).resolve().parent / 'data_jnu'

    if readfunc == read_jnu_ap:
        x_s1, y_s1 = readfunc(str(path / f'{src1}/'), sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_s1, y_s1 = readfunc(str(path / f'{src1}/'), sample_len, nspc, norm='first')
    x_s1 = torch.tensor(x_s1, dtype=torch.float)[:, 0:1, :]
    y_s1 = torch.tensor(y_s1, dtype=torch.long)

    if readfunc == read_jnu_ap:
        x_s2, y_s2 = readfunc(str(path / f'{src2}/'), sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_s2, y_s2 = readfunc(str(path / f'{src2}/'), sample_len, nspc, norm='first')
    x_s2 = torch.tensor(x_s2, dtype=torch.float)[:, 0:1, :]
    y_s2 = torch.tensor(y_s2, dtype=torch.long)

    if union:
        x_s = torch.cat([x_s1, x_s2], 0)
        y_s = torch.cat([y_s1, y_s2], 0)

    if readfunc == read_jnu_ap:
        x_t, y_t = readfunc(str(path / f'{target}/'), sample_len, nspc, norm='first', feature_scaling=feature_scaling)
    else:
        x_t, y_t = readfunc(str(path / f'{target}/'), sample_len, nspc, norm='first')
    x_t = torch.tensor(x_t, dtype=torch.float)[:, 0:1, :]
    y_t = torch.tensor(y_t, dtype=torch.long)

    x = [[] for _ in range(10)]
    y = [[] for _ in range(10)]
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
