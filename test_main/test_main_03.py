import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # 加载数据
    file_path = '/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/test_main/IAutoRec_filmTrust_1.csv'  # 替换为你的文件路径
    data = pd.read_csv(file_path)

    # 数据处理
    data['Result Prefix'] = data['Result'].apply(lambda x: x.split('_')[0])

    # 筛选目标 ID
    data_1884 = data[data['Result'].str.contains('1884')]
    data_1734 = data[data['Result'].str.contains('1734')]

    # 生成图表
    for data_subset, target_id in zip([data_1884, data_1734], [1884, 1734]):
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Result Prefix', y='Composite Impact Score', data=data_subset)
        plt.title(f'Composite Impact Score for Result {target_id}')
        plt.xticks(rotation=45)

        # 调整Y轴的刻度范围
        y_min = data_subset['Composite Impact Score'].min() * 0.65  # 例如，将最小值减少5%
        y_max = data_subset['Composite Impact Score'].max() * 1.35  # 例如，将最大值增加5%
        plt.ylim([y_min, y_max])

        # 在每个条形上添加数值注释
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.4f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),
                        textcoords='offset points')
        plt.show()
