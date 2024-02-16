import numpy as np

if __name__ == '__main__':
    # 读取数据
    data = []
    with open('/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/data/Ratings.txt', 'r') as file:
        for line in file:
            line = line.strip().split()
            uid, iid, rating = int(line[0]), int(line[1]), float(line[2])
            data.append((uid, iid, rating))

    # 统计每个uid对应的iid数量
    uid_iid_count = {}
    for uid, iid, rating in data:
        if uid in uid_iid_count:
            uid_iid_count[uid].add(iid)
        else:
            uid_iid_count[uid] = {iid}

    # 找到最多和最少iid数量
    max_iid_count = max(len(iids) for iids in uid_iid_count.values())
    min_iid_count = min(len(iids) for iids in uid_iid_count.values())

    # print("最多iid数量:", max_iid_count)
    # print("最少iid数量:", min_iid_count)

    # 计算每个uid对应的iid数量
    uid_iid_counts = [len(iids) for iids in uid_iid_count.values()]

    # 计算中位数和平均数
    median_iid_count = np.median(uid_iid_counts)
    mean_iid_count = np.mean(uid_iid_counts)
    # 计算标准差
    std_iid_count = np.std(uid_iid_counts)
    print("Standard deviation of iid count:", std_iid_count)

    # 判断数据分布情况并选择指标
    if std_iid_count < 1:
        indicator = "both median and mean are close, either can be used"
    elif mean_iid_count > median_iid_count:
        print("Indicator use median:", median_iid_count)
    else:
        print("Indicator use mean:", mean_iid_count)





