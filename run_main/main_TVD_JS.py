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
    # 四舍五入ratings，改0评分为1
    df['rating'] = df['rating'].round().astype(int)
    df['rating'] = df['rating'].replace(0, 1)

    # 计算每个iid不同ratings的概率
    item_rating_probability = df.pivot_table(index='iid', columns='rating', aggfunc='size', fill_value=0)

    # 添加缺失的评分列（如果有）
    for rating in all_ratings:
        if rating not in item_rating_probability:
            item_rating_probability[rating] = 0

    # 将计数转换为概率
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
    basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/AUSH/data/experimental_data/1_target_ids/"

    real_dat = '/Users/wangkai/PycharmProjects/ShillingAttack/AUSH/data/data/3_old/filmTrust_train.dat'

    fake_dat_list = ['filmTrust_1282_average_50_36.dat','filmTrust_1282_bandwagon_50_36.dat',
                'filmTrust_1282_BigGan_50_36.dat','filmTrust_1282_gan_50_36.dat',
                'filmTrust_1282_random_50_36.dat','filmTrust_1282_segment_50_36.dat']

    real_df = read_data(real_dat)
    real_item_rating_probability = get_item_rating_probability(real_df, [1,2,3,4,5])

    for fake_dat in fake_dat_list:
        if fake_dat.split('_')[1] == '1282':

            fake_df = read_data(basic_path + fake_dat)
            fake_df = filter_fake_data_by_real(real_df, fake_df)
            item_rating_probability = get_item_rating_probability(fake_df, [1,2,3,4,5])

            # 计算TVD和JS散度
            tvd_values = tvd(real_item_rating_probability.values, item_rating_probability.values)
            js_values = js_divergence(real_item_rating_probability.values, item_rating_probability.values)

            # 计算平均TVD和JS散度
            avg_tvd = np.mean(tvd_values)
            avg_js = np.mean(js_values)

            print(fake_dat)
            print("avg_tvd : ",avg_tvd)
            print("avg_js : ", avg_js)

