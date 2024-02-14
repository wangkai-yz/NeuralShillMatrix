import os
import pandas as pd
import matplotlib.pyplot as plt

def read_data(file_path):
    """
    读取.dat文件并返回DataFrame。
    """
    data = pd.read_csv(file_path, sep='\s+', header=None)
    columns = ['uid', 'pred_rating', 'hit_rate_1', 'hit_rate_3', 'hit_rate_5', 'hit_rate_10', 'hit_rate_20', 'hit_rate_50', 'rank']
    data.columns = columns
    return data

def calculate_hit_rate_proportion(data):
    """
    计算命中率为1的uid占总数的比例。
    """
    hit_rate_columns = ['hit_rate_1', 'hit_rate_3', 'hit_rate_5', 'hit_rate_10', 'hit_rate_20', 'hit_rate_50']
    total_uids = len(data)
    hit_uids = data[hit_rate_columns].any(axis=1).sum()
    return hit_uids / total_uids


def plotting_chart(attack_methods, proportions):
    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(attack_methods, proportions, color='skyblue')
    # plt.xlabel('The Attack Methods are based on NNMF')
    plt.xlabel('The Attack Methods are based on IAutoRec')
    plt.ylabel('Hit Rate Proportion (%)')
    plt.title('Hit Rate Proportion by Attack Method for IAutoRec_1734')
    # plt.title('Hit Rate Proportion by Attack Method for NNMF_1734')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)

    # 在每个柱子上方添加数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.2f}%", ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--')

    # 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    filmTrust = "/Users/wangkai/PycharmProjects/ShillingAttack/AUSH/result/experimental_result/IAutoRec_filmTrust/1734/1_target_ids"
    # filmTrust = "/Users/wangkai/PycharmProjects/ShillingAttack/AUSH/result/experimental_result/NNMF_filmTrust/1734/1_target_ids"
    hit_rate_proportions = []
    attack_methods = []

    for filename in os.listdir(filmTrust):
        file_path = os.path.join(filmTrust, filename)
        data = read_data(file_path)
        proportion = calculate_hit_rate_proportion(data)
        hit_rate_proportions.append(proportion * 100)
        attack = str(filename).split('_')[3]
        attack_methods.append(attack)
        print(f"{attack}: {proportion:.2%}")

    plotting_chart(attack_methods, hit_rate_proportions)

