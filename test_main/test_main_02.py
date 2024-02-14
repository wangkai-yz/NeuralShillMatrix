import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == '__main__':


    # 数据准备
    data = {
        "Category": ["average", "bandwagon", "BigGan", "gan", "random", "segment"],
        "avg_tvd": [0.04932529369126076, 0.16285451890247438, 0.010299398370763942,
                    0.007228965851784586, 0.1640102394402011, 0.20944917962242066],
        "avg_js": [0.01186978862276427, 0.06316858577693088, 0.0009672312611710702,
                   0.0007606415278430077, 0.06350772175509979, 0.08532777387383897]
    }
    df = pd.DataFrame(data)

    # 绘图
    plt.figure(figsize=(12, 6))

    # avg_tvd
    plt.subplot(1, 2, 1)
    sns.barplot(x='Category', y='avg_tvd', data=df)
    plt.title('Average TVD by Category')
    plt.xticks(rotation=45)

    # avg_js
    plt.subplot(1, 2, 2)
    sns.barplot(x='Category', y='avg_js', data=df)
    plt.title('Average JS by Category')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
