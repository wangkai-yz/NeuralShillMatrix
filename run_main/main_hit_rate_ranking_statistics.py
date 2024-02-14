import os
import re
import pandas as pd

def calculate_rank_changes_modified(pre_attack_data, post_attack_data):
    """
    计算pred_rating和rank的差值的平均数和方差。
    """
    # 将字符串转换为数值，对于rank，将None替换为51
    pre_attack_data['pred_rating'] = pre_attack_data['pred_rating'].astype(float)
    post_attack_data['pred_rating'] = post_attack_data['pred_rating'].astype(float)
    pre_attack_data['rank'] = pre_attack_data['rank'].replace('None', 51).astype(float)
    post_attack_data['rank'] = post_attack_data['rank'].replace('None', 51).astype(float)

    # 过滤掉仅存在于post_attack_data中的uid
    common_uids = pre_attack_data['uid'].unique()
    filtered_post_attack_data = post_attack_data[post_attack_data['uid'].isin(common_uids)]

    # 使用map函数计算差值
    pred_rating_diff = filtered_post_attack_data.set_index('uid')['pred_rating'] - pre_attack_data.set_index('uid')['pred_rating']
    rank_diff = pre_attack_data.set_index('uid')['rank'] - filtered_post_attack_data.set_index('uid')['rank']

    # 计算平均数、方差和标准差
    avg_pred_rating_diff = pred_rating_diff.mean()
    var_pred_rating_diff = pred_rating_diff.var()
    std_pred_rating_diff = pred_rating_diff.std()
    avg_rank_diff = rank_diff.mean()
    var_rank_diff = rank_diff.var()
    std_rank_diff = rank_diff.std()

    return avg_pred_rating_diff, var_pred_rating_diff, std_pred_rating_diff, avg_rank_diff, var_rank_diff, std_rank_diff


def write_rank_changes_to_file(rank_changes, file_path):
    """
    将排名变化的数据写入到文件中。
    """
    # 将rank_changes转换为DataFrame（如果它还不是）
    if not isinstance(rank_changes, pd.DataFrame):
        rank_changes = pd.DataFrame(rank_changes)

    # 写入文件
    rank_changes.to_csv(file_path, index=False)


def read_data(file_path):
    """
    读取.dat文件并返回DataFrame。
    假定数据的格式如上所述，以制表符或空格分隔。
    """
    # 读取数据
    data = pd.read_csv(file_path, sep='\s+', header=None)

    columns = ['uid', 'pred_rating', 'hit_rate_1', 'hit_rate_3', 'hit_rate_5',
               'hit_rate_10', 'hit_rate_20', 'hit_rate_50', 'rank']
    data.columns = columns

    return data

def filter_fake_data_by_real(real_df, fake_df):
    """
    删除fake_df中不存在于real_df的uid。
    """
    common_uid = real_df['uid'].unique()
    filtered_fake_df = fake_df[fake_df['uid'].isin(common_uid)]
    return filtered_fake_df

def calculate_impact(pre_attack_data, post_attack_data, alpha=0.5, beta=0.3, gamma=0.2):
    # 计算排名变化
    rank_change = pre_attack_data['rank'].replace('None', 51).fillna(51).astype(float) - post_attack_data['rank'].replace('None', 51).fillna(51).astype(float)
    avg_rank_change = rank_change.mean()

    # 计算评分变化
    rating_change = post_attack_data['pred_rating'] - pre_attack_data['pred_rating']
    avg_rating_change = rating_change.mean()

    # 计算命中率变化
    hit_rate_columns = ['hit_rate_1', 'hit_rate_3', 'hit_rate_5', 'hit_rate_10', 'hit_rate_20', 'hit_rate_50']
    hit_rate_change = post_attack_data[hit_rate_columns] - pre_attack_data[hit_rate_columns]
    avg_hit_rate_change = hit_rate_change.mean().mean()

    # 综合影响分数
    composite_impact_score = alpha * avg_rank_change + beta * avg_rating_change + gamma * avg_hit_rate_change
    return composite_impact_score

def count_unique_hit_rates(data):
    """
    计算不同命中率级别的唯一数量。
    """
    hit_rate_columns = ['hit_rate_1', 'hit_rate_3', 'hit_rate_5', 'hit_rate_10', 'hit_rate_20', 'hit_rate_50']

    # 初始化计数器
    hit_rate_counts = {col: 0 for col in hit_rate_columns}

    # 对于每一行数据，检查每个命中率级别是否为1，如果是，增加对应的计数
    for index, row in data.iterrows():
        for col in hit_rate_columns:
            if row[col] == 1:
                # 检查更高的命中率是否为0，如果是，则这是独立的命中率
                if all(row[higher_col] == 0 for higher_col in hit_rate_columns if higher_col > col):
                    hit_rate_counts[col] += 1
                break  # 不需要检查更低的命中率

    return hit_rate_counts

def count_full_hit_rates(data):
    """
    计算完全命中（所有命中率为1）的数量。
    """
    hit_rate_columns = ['hit_rate_1', 'hit_rate_3', 'hit_rate_5', 'hit_rate_10', 'hit_rate_20', 'hit_rate_50']
    # 检查所有命中率列是否都为1
    full_hits = data[hit_rate_columns].all(axis=1).sum()
    return full_hits

def generate_according_by_attack_mode(basic_path, target_ids, attack_together_num, attack_name):
    # 创建一个空的数据框来存储结果
    columns = ["Result", "Avg Pred Rating Diff", "Var Pred Rating Diff", "Std Pred Rating Diff", "Avg Rank Diff",
               "Var Rank Diff", "Std Rank Diff"]
    result_df = pd.DataFrame(columns=columns)

    for target_id in target_ids:
        basic_str = "/" + str(target_id) + "/IAutoRec_filmTrust_" + str(target_id)
        pre_attack_data = read_data(basic_path + basic_str)

        for attack_together in attack_together_num:
            sub_path = f'/{target_id}/{attack_together}'
            if attack_together == '1_target_ids':
                for root, dirs, files in os.walk(basic_path + sub_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        post_attack_data = read_data(file_path)
                        post_attack_data = filter_fake_data_by_real(pre_attack_data, post_attack_data)

                        # if file_path.split('_')[-3] in attack_name:
                        result = file_path.split('_')[-3] + '_' + str(target_id) + '_' + attack_together

                        # 计算综合影响分数
                        impact_score = calculate_impact(pre_attack_data, post_attack_data)

                        avg_pred_rating_diff, var_pred_rating_diff, std_pred_rating_diff, avg_rank_diff, var_rank_diff, std_rank_diff = (
                            calculate_rank_changes_modified(pre_attack_data, post_attack_data))

                        # 创建一个临时数据框来存储当前结果
                        temp_df = pd.DataFrame({
                            "Result": [result],
                            "Avg Pred Rating Diff": [avg_pred_rating_diff],
                            "Var Pred Rating Diff": [var_pred_rating_diff],
                            "Std Pred Rating Diff": [std_pred_rating_diff],
                            "Avg Rank Diff": [avg_rank_diff],
                            "Var Rank Diff": [var_rank_diff],
                            "Std Rank Diff": [std_rank_diff],
                            "Composite Impact Score": [impact_score]
                        })

                        # 使用concat方法将临时数据框与结果数据框合并
                        result_df = pd.concat([result_df, temp_df], ignore_index=True)


    # 将数据框保存为CSV文件
    result_df.to_csv(f"IAutoRec_filmTrust_1.csv", index=False)

def test_001(basic_path, target_ids, attack_together_num, attack_name):
    for target_id in target_ids:
        basic_str = "/" + str(target_id) + "/IAutoRec_filmTrust_" + str(target_id)
        pre_attack_data = read_data(basic_path + basic_str)

        for attack_together in attack_together_num:
            sub_path = f'/{target_id}/{attack_together}'
            if attack_together == '1_target_ids':
                for root, dirs, files in os.walk(basic_path + sub_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        post_attack_data = read_data(file_path)
                        post_attack_data = filter_fake_data_by_real(pre_attack_data, post_attack_data)

                        print(file_path, count_full_hit_rates(post_attack_data))


if __name__ == '__main__':

    basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/AUSH/result/experimental_result/IAutoRec_filmTrust"
    # target_ids = [1282, 1734]
    target_ids = [1884,1734]
    attack_together_num = ['1_target_ids', '2_target_ids', '3_target_ids']
    attack_list = ['BigGan','gan','segment','average','random','bandwagon']
    # attack_list = ['bandwagon']

    # generate_according_by_attack_mode(basic_path, target_ids, attack_together_num, attack_list)
    test_001(basic_path, target_ids, attack_together_num, attack_list)



