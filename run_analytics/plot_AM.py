import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_data(file_path):
    """
    读取CSV文件并返回一个DataFrame。
    Read a CSV file and return a DataFrame.
    """
    data = pd.read_csv(file_path)
    # Convert 'Composite Impact Score' to numeric, ensuring all values are floats
    # data['Composite Impact Score'] = pd.to_numeric(data['Composite Impact Score'], errors='coerce')
    return data

def plotting_chart(x_name, y_name, file, title, output_path, image_name, data):
    """
    根据给定的数据绘制条形图。
    Plot a bar chart based on the given data.

    参数说明 Parameters:
    - x_name: X轴数据名称 X-axis data name
    - y_name: Y轴数据名称 Y-axis data name
    - file: 文件名，用于错误信息 Filename, used for error messages
    - title: 图表标题 Chart title
    - output_path: 图表保存路径 Output path for the chart
    - image_name: 保存的图像名称 Saved image name
    - data: 数据源 Data source for the chart
    """

    if data[x_name].dtype in ['float64', 'int64']:

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

    else:
        print("Error: 'Composite Impact Score' is not numeric for file:", file)

def plotting_AM(files, cb_path):
    """
    从给定的文件中绘制“攻击模型”图表。
    Plot "Attack Model" charts from the given files.

    参数说明 Parameters:
    - files: 文件列表 List of files to process
    - cb_path: 文件所在的基本路径 The base path where files are located
    """
    image_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/image/AM"

    # 筛选以"AM"开头的文件
    selected_files = [file for file in files if file.startswith("AM")]

    for file_name in selected_files:
        if file_name.split('_')[1] != 'MT' and file_name.split('_')[1] != 'NNMF':
            file_path = os.path.join(cb_path, file_name)
            data = read_data(file_path)

            model_name = file_name.split('_')[1]
            target_id = file_name.split('_')[3]
            target_id = target_id.split('.')[0]

            # plotting_chart('Composite Impact Score', 'Attack Type', file_name, f'Composite Impact Score by Attack Type for {model_name} {target_id}',image_path, file_name.replace('.csv', '.jpg').replace('filmTrust',f'(Composite Impact Score)'), data)
            plotting_chart('Ratio Rank 1', 'Attack Type', file_name, f'Ratio Rank 1 by Attack Type for {model_name} {target_id}', image_path, file_name.replace('.csv', '.jpg').replace('filmTrust',f'(Ratio Rank 1)'), data)

def plotting_AM_MT(files, cb_path):
    """
    为“AM_MT”攻击模型绘制图表。
    Plot charts for "AM_MT" attack models.

    参数说明 Parameters:
    - files: 文件列表 List of files to process
    - cb_path: 文件所在的基本路径 The base path where files are located
    """
    image_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/image/AM_MT"

    # 筛选以"AM"开头的文件
    selected_files = [file for file in files if file.startswith("AM")]

    for file_name in selected_files:
        if file_name.split('_')[1] == 'MT' and file_name.split('_')[2] != 'NNMF':
            file_path = os.path.join(cb_path, file_name)
            data = read_data(file_path)

            model_name = file_name.split('_')[1]
            target_id = file_name.split('_')[3]
            target_id = target_id.split('.')[0]

            if target_id == '2001':
                plotting_chart('Composite Impact Score', 'Attack Number', file_name,
                               f'Composite Impact Score by Attack Type for {model_name} {target_id}',
                               image_path, file_name.replace('.csv', '(Composite Impact Score).jpg'), data)
                plotting_chart('Ratio Rank 1', 'Attack Number', file_name,
                               f'Ratio Rank 1 by Attack Type for {model_name} {target_id}',
                               image_path, file_name.replace('.csv', '(Ratio Rank 1).jpg'), data)

def plotting_DC(files, cb_path):
    """
    为“DC”攻击模型绘制图表。
    Plot charts for "DC" attack models.

    参数说明 Parameters:
    - files: 文件列表 List of files to process
    - cb_path: 文件所在的基本路径 The base path where files are located
    """
    image_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/image/DC"

    # 筛选以"AM"开头的文件
    selected_files = [file for file in files if file.startswith("DC")]

    for file_name in selected_files:
        if file_name.split('_')[1] != 'MT' and file_name.split('_')[1] != 'NNMF':
            file_path = os.path.join(cb_path, file_name)
            data = read_data(file_path)

            target_id = file_name.split('_')[2]
            target_id = target_id.split('.')[0]

            plotting_chart('Average Total Variation Gap', 'Attack Type', file_name, f'Average Total Variation Gap by Attack Type for {target_id}',
                           image_path, file_name.replace('.csv', '.jpg').replace('filmTrust',f'(Average Total Variation Gap)'), data)
            plotting_chart('Average JS Divergence', 'Attack Type', file_name, f'Average JS Divergence by Attack Type for {target_id}',
                           image_path, file_name.replace('.csv', '.jpg').replace('filmTrust',f'(Average JS Divergence)'), data)

if __name__ == '__main__':
    csv_basic_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/result/csv_evaluation"
    # 获取目录下所有文件
    all_files = os.listdir(csv_basic_path)

    plotting_AM_MT(all_files, csv_basic_path)






