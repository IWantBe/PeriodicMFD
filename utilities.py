import os
import random
import numpy as np
import torch
import time
import torch.nn as nn


def seed_everything(seed, strengthen=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if strengthen:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ['PYTHONHASHSEED'] = str(seed)


def timer(func):
    '''timer'''

    def wrapper(*args, **kwargs):
        t1 = time.time()
        ret = func(*args, **kwargs)
        t2 = time.time()
        t = int(t2 - t1)
        ss = t % 60
        t //= 60
        mm = t % 60
        hh = t // 60
        print(f'\ntotal time: {hh:02}:{mm:02}:{ss:02}\n')
        return ret

    return wrapper


@timer
def run_EX(tasks: list, funcs: list, num: int):
    all_res = []
    for i in range(num):
        res = []
        for task in tasks:
            print(f'\nNo. {i+1}/{num}, now task{task}')
            li = [func(*task) for func in funcs]
            res.append(li)
        all_res.append(torch.tensor(res))

    for i, mat in enumerate(all_res):
        print(f'\nNo. {i+1}')
        for row in mat:
            for v in row:
                print(f'{v.item():.5f} ', end='')
            print()
    mean = sum(all_res) / len(all_res)
    print(f'\nmean')
    for row in mean:
        for v in row:
            print(f'{v.item():.5f} ', end='')
        print()


def get_params_num(model: nn.Module) -> int:
    '''Calculate the number of model parameters'''
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    return k
