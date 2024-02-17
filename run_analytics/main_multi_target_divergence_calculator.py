import os
import re
import numpy as np
import pandas as pd
import scipy.stats
def read_data(file_path):
    """
    读取.dat文件并返回DataFrame。
    假定数据的格式如上所述，以制表符或空格分隔。
    """
    # 读取数据
    data = pd.read_csv(file_path, sep='\s+', header=None)

    columns = ['uid', 'iid', 'rating']
    data.columns = columns

    return data

def get_item_rating_probability(df, all_ratings):
    # Round off the ratings, changing 0 to 1
    df['rating'] = df['rating'].round().astype(int)
    df['rating'] = df['rating'].replace(0, 1)

    # The probability of different ratings for each iid is calculated
    item_rating_probability = df.pivot_table(index='iid', columns='rating', aggfunc='size', fill_value=0)

    # Add the missing rating column (if any)
    for rating in all_ratings:
        if rating not in item_rating_probability:
            item_rating_probability[rating] = 0

    # Convert the counts to probabilities
    item_rating_probability = item_rating_probability.div(item_rating_probability.sum(axis=1), axis=0)
    return item_rating_probability

def tvd(p, q):
    """计算总变差距离 (TVD)"""
    return 0.5 * np.sum(np.abs(p - q), axis=1)

def kl_divergence(p, q):
    """计算Kullback-Leibler散度"""
    # 避免除以零，使用小数值替换零
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    return np.sum(p * np.log(p / q), axis=1)

def js_divergence(p, q):
    """计算JS散度 (JS)"""
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

def filter_fake_data_by_real(real_df, fake_df):
    """
    删除fake_df中不存在于real_df的iid。
    """
    common_iids = real_df['iid'].unique()
    filtered_fake_df = fake_df[fake_df['iid'].isin(common_iids)]
    return filtered_fake_df

if __name__ == '__main__':

    real_basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/data/raw_data"
    fake_basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/data/merged_data"
    csv_basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/csv_evaluation"

    target_ids = [1689,2001]
    attack_list = ['BigGan','gan','segment','average','random','bandwagon']

    real_data_path = real_basic_path + '/filmTrust_train.dat'
    real_df = read_data(real_data_path)
    real_item_rating_probability = get_item_rating_probability(real_df, [1,2,3,4,5])

    columns = ["Target Id", "Attack Type", "Attack Num", "Average Total Variation Gap", "Average JS Divergence"]

    for target_id in target_ids:
        for attack_type in attack_list:
            result_df = pd.DataFrame(columns=columns)
            for attack_num in range(5):

                fake_data_path = '_'.join(['filmTrust', str(target_id), attack_type, '50', '16']) + '.dat'
                fake_data_path = os.path.join(fake_basic_path + f'/{target_id}' + f'/{attack_num + 1}', fake_data_path)

                fake_df = read_data(fake_data_path)
                fake_df = filter_fake_data_by_real(real_df, fake_df)
                item_rating_probability = get_item_rating_probability(fake_df, [1, 2, 3, 4, 5])

                # 计算TVD和JS散度
                tvd_values = tvd(real_item_rating_probability.values, item_rating_probability.values)
                js_values = js_divergence(real_item_rating_probability.values, item_rating_probability.values)

                # 计算平均TVD和JS散度
                avg_tvd = np.mean(tvd_values)
                avg_js = np.mean(js_values)

                print(f"{'_'.join(['DC_MT', str(target_id), attack_type, str(attack_num + 1)])} avg_tvd: {avg_tvd} avg_js: {avg_js}")
                # 创建一个临时数据框来存储当前结果
                temp_df = pd.DataFrame({
                    "Target Id": [target_id],
                    "Attack Type": [attack_type],
                    "Attack Num": [attack_num + 1],
                    "Average Total Variation Gap": [avg_tvd],
                    "Average JS Divergence": [avg_js]
                })

                # 使用concat方法将临时数据框与结果数据框合并
                result_df = pd.concat([result_df, temp_df], ignore_index=True)

            # 将数据框保存为CSV文件
            result_df.to_csv(os.path.join(csv_basic_path, f"{'_'.join(['DC_MT', 'filmTrust', str(target_id), attack_type])}.csv"),index=False)
