import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

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

def test_01(basic_path):
    # 获取目录下所有文件 /AM_MT_IAutoRec_2001_segment.csv
    all_files = os.listdir(basic_path)
    selected_files = [file for file in all_files if file.startswith("AM_MT_NNMF_2001")]

    columns = ["Attack_Type", "slope", "r_value ** 2"]
    result_df = pd.DataFrame(columns=columns)
    for one in selected_files:
        file_path = os.path.join(basic_path, one)
        data = pd.read_csv(file_path)

        Attack_Type = one.split('_')[4]
        Attack_Type = Attack_Type.split('.')[0]

        # Performing linear regression analysis between Attack Number and Ratio Rank 1
        slope, intercept, r_value, p_value, std_err = linregress(data['Attack Number'], data['Ratio Rank 1'])
        # Output the results of the linear regression
        print(Attack_Type, slope, r_value ** 2)

        # 创建一个临时数据框来存储当前结果
        temp_df = pd.DataFrame({
            "Attack_Type": [Attack_Type],
            "slope": [slope],
            "r_value ** 2": [r_value ** 2]
        })

        # 使用concat方法将临时数据框与结果数据框合并
        result_df = pd.concat([result_df, temp_df], ignore_index=True)
    # 将数据框保存为CSV文件
    result_df.to_csv(os.path.join(basic_path, "NNMF_2001.csv"), index=False)

def plotting_chart(x_name, y_name, output_path, image_name, title, data):
    # Set global font sizes
    plt.rcParams['axes.titlesize'] = 18  # Title font size
    plt.rcParams['axes.labelsize'] = 18  # Axis labels font size
    plt.rcParams['xtick.labelsize'] = 18  # X-axis tick labels font size
    plt.rcParams['ytick.labelsize'] = 18  # Y-axis tick labels font size
    plt.rcParams['legend.fontsize'] = 18  # Legend font size
    plt.rcParams['font.size'] = 18  # Global font size for text in plots

    plt.figure(figsize=(10, 6))
    # , color="skyblue"
    ax = sns.barplot(x=y_name, y=x_name, data=data)

    plt.title(title)
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')
    plt.tight_layout()
    # plt.show()
    output_path = os.path.join(output_path, image_name)
    print(output_path)
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory

if __name__ == '__main__':

    basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/csv_evaluation"
    image_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/image/AM_MT"
    file_list = ['IAutoRec_2001.csv','NNMF_2001.csv']
    for file in file_list:
        file_path = os.path.join(basic_path, file)
        data = pd.read_csv(file_path)
        plotting_chart('r_value ** 2', 'Attack_Type', image_path, file.replace('.csv', '.jpg'),
                       file.replace('_','  ').replace('.csv','  r_value ** 2'), data)