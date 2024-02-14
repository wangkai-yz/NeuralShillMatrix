import numpy as np
import os

data_dir = "../data/raw_data/"
def read_npy():
    # 读取 filmTrust_id_mapping.npy 文件
    data = np.load('..\\data\\attack_data\\filmTrust_50_36_1808_attackSetting.npy', allow_pickle=True)

    # 查看数据
    for dictionary in data:
        for i in dictionary:
            print(len(i))

def read_path(dataset_name):
    testing_path = os.path.join(data_dir, f"{dataset_name}_test.dat")

    print(testing_path)

if __name__ == '__main__':
    read_npy()