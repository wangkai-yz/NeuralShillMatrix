import os
import gzip
import itertools
from NeuralShillMatrix.utils.data_loader import *
from sklearn.model_selection import train_test_split

data_dir = "../data/raw_data/"
def read_gzip_file(path):
    """
    Generator that yields JSON objects from a gzip-compressed file.
    生成器，从gzip压缩文件中生成JSON对象。
    """
    with gzip.open(path, 'rb') as file:
        for line in file:
            yield eval(line)

def convert_to_dataframe(path):
    """
    Converts a list of dictionaries into a pandas DataFrame.
    将字典列表转换为pandas数据帧。
    """
    index = 0
    data_dict = {}
    for data in read_gzip_file(path):
        data_dict[index] = data
        index += 1
    return pd.DataFrame.from_dict(data_dict, orient='index')

def preprocess_dataset(dataset_name, file_path):
    """
    Preprocesses the dataset by loading, transforming, and splitting it into training and testing sets.
    This function loads the dataset from a given file path, assigns unique numeric IDs to users and items,
    and splits the data into training and testing sets.
    通过加载、转换和划分训练集和测试集对数据集进行预处理。
    该函数从给定的文件路径加载数据集，为用户和物品分配唯一的数字id，
    并将数据分为训练集和测试集。

    Parameters:
    - dataset_name: The name of the dataset for naming output files.
    - file_path: The path to the dataset file.
    参数:
    —dataset_name:用于命名输出文件的数据集名称。
    —file_path:数据集文件的路径。
    """
    # Load the dataset
    data = pd.read_csv(file_path, sep=' ', header=None, names=['user_id', 'item_id', 'rating'])

    # Mapping unique users and items to numeric IDs
    user_mapping = {uid: num for num, uid in enumerate(data['user_id'].unique())}
    item_mapping = {iid: num for num, iid in enumerate(data['item_id'].unique())}

    # Applying mappings to the dataset
    data['user_id'] = data['user_id'].map(user_mapping)
    data['item_id'] = data['item_id'].map(item_mapping)

    # Reporting dataset statistics
    total_users, total_items, total_ratings = len(user_mapping), len(item_mapping), len(data)
    print(f'User count: {total_users} \tItem count: {total_items} \tRating count: {total_ratings} \tSparsity: {total_ratings / (total_items * total_users):.4f}')
    print(f'Average ratings per user: {total_ratings / total_users:.2f}')

    # Splitting the dataset into training and testing sets
    train_indices, test_indices = train_test_split(data.index, test_size=0.1, random_state=42)
    training_data = data.loc[train_indices]
    testing_data = data.loc[test_indices]

    # Saving the processed datasets and mappings
    training_path = os.path.join(data_dir, f"{dataset_name}_train.dat")
    testing_path = os.path.join(data_dir, f"{dataset_name}_test.dat")
    training_data.to_csv(training_path, index=False, header=None, sep='\t')
    testing_data.to_csv(testing_path, index=False, header=None, sep='\t')
    np.save(os.path.join(data_dir, f"{dataset_name}_id_mapping.npy"), [user_mapping, item_mapping])

def select_experiment(dataset_name, selection_number, target_user_count):
    """
    Selects target items and users for the experiment based on item popularity and rating threshold.
    基于项目流行度和评分阈值选择目标项目和用户进行实验。
    """
    # 加载训练数据和测试数据
    test_data_path = os.path.join(data_dir, f"{dataset_name}_test.dat")
    train_data_path = os.path.join(data_dir, f"{dataset_name}_train.dat")
    data_loader = DataLoader(path_to_train_data=train_data_path, path_to_test_data=test_data_path,
                             file_header=['user_id', 'item_id', 'rating'],delimiter='\t', enable_logging=True)

    # 获取最受欢迎的前selection_number个项目
    item_popularity = data_loader.calculate_item_popularity()
    popular_items_sorted = np.array(item_popularity).argsort()[::-1]
    selected_bandwagon = popular_items_sorted[:selection_number]
    print('Selected bandwagon items:', selected_bandwagon)

    # 测试数据的平均评分，大于等于3则赋值3.0，最后得到等级阈值。
    average_rating_threshold = data_loader.test_data.rating.mean()
    average_rating_threshold = average_rating_threshold if average_rating_threshold < 3 else 3.0
    print('Rating threshold:', average_rating_threshold)

    # 获取最受欢迎的前20个项目，随机选取任意的selection_number个进行组合，作为关联项目。
    candidate_items = popular_items_sorted[:20]
    candidate_combinations = list(itertools.combinations(candidate_items, selection_number))

    # 将受欢迎的项目划分，然后选择每个部分中的前2个，也就是16个。从评分次数为3的项目中随机选择4个项目，组合在一起成为备选的目标项目。
    selection_results = {}
    target_items = [j for i in range(2, 10) for j in popular_items_sorted[i * len(popular_items_sorted) // 10:(i * len(popular_items_sorted) // 10) + 2]][::-1]
    random_items = list(np.random.choice([i for i in range(len(item_popularity)) if item_popularity[i] == 3], 4, replace=False))
    target_items = random_items + target_items

    print('Target items:', target_items)
    print('Ratings count for target items:', [item_popularity[i] for i in target_items])

    for target in target_items:
        # 找到目标项被评价过的用户
        target_rated = set(data_loader.train_data[data_loader.train_data.item_id == target].user_id.values)
        # 创建数据副本，排除目标项被评价过的用户，并且评分高于阈值的样本
        data_tmp = data_loader.train_data[~data_loader.train_data.user_id.isin(target_rated)].copy()
        data_tmp = data_tmp[data_tmp.rating >= average_rating_threshold]
        # 随机打乱候选项组合的顺序
        np.random.shuffle(candidate_combinations)

        # 用于标记是否找到了目标用户
        found_target_users = False
        # 遍历候选项组合
        for selected_items in candidate_combinations:
            # 计算候选项组合中的用户数量
            target_users = data_tmp[data_tmp.item_id.isin(selected_items)].groupby('user_id').size()
            if target_users[(target_users == selection_number)].shape[0] >= target_user_count:
                # 如果满足目标用户数量的条件，则记录选择结果并打印目标项
                target_users = sorted(target_users[(target_users == selection_number)].index)
                selection_results[target] = [sorted(selected_items), target_users]
                print('target:', target)
                found_target_users = True
                break

        # 如果未找到目标用户，则采取备用策略
        if not found_target_users:
            for selected_items in candidate_combinations:
                target_users = data_tmp[data_tmp.item_id.isin(selected_items)].groupby('user_id').size()
                target_users = sorted(dict(target_users).items(), key=lambda x: x[1], reverse=True)
                min_users = target_users[target_user_count][1]
                target_users = [i[0] for i in target_users[:target_user_count] if i[1] > selection_number // 2]
                if len(target_users) >= target_user_count:
                    selection_results[target] = [sorted(selected_items), sorted(target_users)]
                    print('target:', target, 'min rated selected item num：', min_users)
                    found_target_users = True
                    break

        if not found_target_users:
            print('target:', target, 'non-targeted user')

    key = list(selection_results.keys())
    selected_items = [','.join(map(str, selection_results[k][0])) for k in key]
    target_users = [','.join(map(str, selection_results[k][1])) for k in key]
    selected_items = pd.DataFrame(dict(zip(['id', 'selected_items'], [key, selected_items])))
    target_users = pd.DataFrame(dict(zip(['id', 'target_users'], [key, target_users])))
    output_kwargs = {'index': False, 'header': None, 'sep': '\t'}
    selected_items.to_csv(os.path.join(data_dir, f"{dataset_name}_selected_items"), **output_kwargs)
    target_users.to_csv(os.path.join(data_dir, f"{dataset_name}_target_users"), **output_kwargs)


if __name__ == '__main__':
    dataset_name = 'filmTrust'
    # gzip_path = "C:\\Users\\surface\\PycharmProjects\\ShillingAttack\\NeuralShillMatrix\\data\\Ratings.txt"
    gzip_path = "/Users/wangkai/PycharmProjects/ShillingAttack/NeuralShillMatrix/data/Ratings.txt"
    preprocess_dataset(dataset_name, gzip_path)

    select_experiment(dataset_name, selection_number=3, target_user_count=30)
