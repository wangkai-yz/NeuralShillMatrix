import os
import re
import pandas as pd

def calculate_rank_changes_modified(pre_attack_data, post_attack_data):
    """
    计算pred_rating和rank的差值的平均数和方差。

    Calculate the mean and variance of the difference between pred_rating and rank.
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

    Write the rank change data to a file.
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

    Read the.dat file and return a DataFrame.
    The data is assumed to be in the format described above, separated by tabs or Spaces.
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

    Remove Uids in fake_df that do not exist in real_df.
    """
    common_uid = real_df['uid'].unique()
    filtered_fake_df = fake_df[fake_df['uid'].isin(common_uid)]
    return filtered_fake_df

def calculate_impact(pre_attack_data, post_attack_data, alpha=0.4, beta=0.3, gamma=0.3):
    """
    计算攻击的综合影响分数。
    Calculate the composite impact score of an attack.
    """
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

def generate_according_by_attack_mode(rm_list, tid_list, a_list, b_path, cb_path):
    """
    根据攻击模式生成评估结果并写入CSV文件。
    Generate evaluation results according to attack modes and write to CSV files.
    """
    columns = ["Model Name", "Target Id", "Attack Type",
               "Avg Pred Rating Diff", "Var Pred Rating Diff","Std Pred Rating Diff",
               "Avg Rank Diff", "Var Rank Diff", "Std Rank Diff",
               "Ratio Rank 1", "Composite Impact Score"]

    for model_name in rm_list:
        for target_id in tid_list:
            real_data_evaluation_name = '_'.join([model_name, 'filmTrust', str(target_id), 'no'])
            real_data_evaluation_path = os.path.join(b_path, real_data_evaluation_name)
            real_data_evaluation = read_data(real_data_evaluation_path)

            result_df = pd.DataFrame(columns=columns)
            for attack_type in a_list:
                fake_data_evaluation_name = '_'.join([model_name, 'filmTrust', str(target_id), attack_type, '50', '16'])
                fake_data_evaluation_path = os.path.join(b_path, fake_data_evaluation_name)
                fake_data_evaluation = read_data(fake_data_evaluation_path)
                fake_data_evaluation = filter_fake_data_by_real(real_data_evaluation, fake_data_evaluation)

                # 计算最后一列值为1的行数，计算比例
                count_rank_1 = fake_data_evaluation[fake_data_evaluation['rank'] == '1'].shape[0]
                total_count = fake_data_evaluation.shape[0]
                ratio_rank_1 = count_rank_1 / total_count

                # 计算综合影响分数
                impact_score = calculate_impact(real_data_evaluation, fake_data_evaluation)
                avg_pred_rating_diff, var_pred_rating_diff, std_pred_rating_diff, avg_rank_diff, var_rank_diff, std_rank_diff = (
                    calculate_rank_changes_modified(real_data_evaluation, fake_data_evaluation))

                # 创建一个临时数据框来存储当前结果
                temp_df = pd.DataFrame({
                    "Model Name": [model_name],
                    "Target Id": [target_id],
                    "Attack Type": [attack_type],
                    "Avg Pred Rating Diff": [avg_pred_rating_diff],
                    "Var Pred Rating Diff": [var_pred_rating_diff],
                    "Std Pred Rating Diff": [std_pred_rating_diff],
                    "Avg Rank Diff": [avg_rank_diff],
                    "Var Rank Diff": [var_rank_diff],
                    "Std Rank Diff": [std_rank_diff],
                    "Ratio Rank 1": [ratio_rank_1],
                    "Composite Impact Score": [impact_score]
                })

                # 使用concat方法将临时数据框与结果数据框合并
                result_df = pd.concat([result_df, temp_df], ignore_index=True)

            # 将数据框保存为CSV文件
            result_df.to_csv(os.path.join(cb_path, f"{'_'.join(['AM', model_name, 'filmTrust', str(target_id)])}.csv"),index=False)

if __name__ == '__main__':

    basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/model_evaluation"
    csv_basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/csv_evaluation"
    target_ids = [1689,1691,1808,1959,2001]
    attack_list = ['BigGan','gan','segment','average','random','bandwagon']
    recommendation_model = ['NNMF','IAutoRec']

    generate_according_by_attack_mode(recommendation_model, target_ids, attack_list, basic_path, csv_basic_path)



