import os
import glob
import pandas as pd


# 定义函数读取文件
def read_dat_file(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, names=['uid', 'iid', 'rating'])


# 定义函数写入数据到新文件
def write_dat_file(data, file_path):
    data.to_csv(file_path, sep='\t', header=False, index=False)

def aggregate_files(base_path, train_file_path):
    method_types = ['average', 'bandwagon', 'BigGan', 'gan', 'random', 'segment']
    project_ids_series = ['1689', '1691', '1808', '1959', '2001']
    # project_ids_series = ['2001', '1959', '1808', '1691', '1689']
    train_data = read_dat_file(train_file_path)

    for method in method_types:
        unique_data = pd.DataFrame()
        accumulated_num = 1
        unique_data_list = {}
        for project_id in project_ids_series:
            # 检查路径是否存在
            path = f"/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/data/merged_data/{project_ids_series[0]}/{accumulated_num}"
            if not os.path.exists(path):
                # 如果路径不存在，则创建
                os.makedirs(path)
            accumulated_pattern = path + f"/filmTrust_{project_ids_series[0]}_{method}_50_16.dat"
            accumulated_num += 1

            pattern = f"{base_path}/filmTrust_{project_id}_{method}_50_16.dat"
            file_data = read_dat_file(pattern)
            # 筛选出存在于当前文件且不存在于train_data中的数据
            unique_to_current = pd.merge(file_data, train_data, on=['uid', 'iid', 'rating'], how='left', indicator=True)
            unique_to_current = unique_to_current[unique_to_current['_merge'] == 'left_only'][['uid', 'iid', 'rating']]

            unique_data = pd.concat([unique_data, unique_to_current], ignore_index=True).drop_duplicates()
            unique_data_list[accumulated_pattern] = unique_data

        # 按照不同的项目ID聚合后的数据写入新文件
        for write_path in unique_data_list.keys():
            # 将train_data添加到聚合数据中
            aggregated_data = pd.concat([train_data, unique_data_list[write_path]], ignore_index=True).drop_duplicates()
            write_dat_file(aggregated_data, write_path)
            print(write_path, len(aggregated_data))

if __name__ == '__main__':
    # 调用主函数
    base_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/data/attack_data"
    train_file_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/data/raw_data/filmTrust_train.dat"
    aggregate_files(base_path, train_file_path)
