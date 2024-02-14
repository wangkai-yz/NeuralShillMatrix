import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

class DataLoader:
    def __init__(self, path_to_train_data, path_to_test_data,
                 file_header=None, delimiter='\t', like_threshold=4, enable_logging=True):
        self.train_data_path = path_to_train_data
        self.test_data_path = path_to_test_data
        self.file_header = file_header if file_header is not None else ['user_id', 'item_id', 'rating']
        self.delimiter = delimiter
        self.like_threshold = like_threshold
        self.enable_logging = enable_logging

        self._initialize_data_loading()

    def _initialize_data_loading(self):
        """
        Initializes the process of loading and processing the data.
        初始化加载和处理数据的过程。
        """
        # Loading the data from files
        self._read_data_files()
        # Converting dataframes to matrices
        self.train_matrix, self.train_matrix_binary = self._convert_to_matrix(self.train_data)
        self.test_matrix, self.test_matrix_binary = self._convert_to_matrix(self.test_data)

    def _read_data_files(self):
        """
        Loads data from the train and test files.
        从训练和测试文件中加载数据。
        """
        if self.enable_logging:
            print("Loading train/test data from: ", self.train_data_path)
        self.train_data = pd.read_csv(self.train_data_path, sep=self.delimiter, names=self.file_header, engine='python').loc[:,
                          ['user_id', 'item_id', 'rating']]
        self.test_data = pd.read_csv(self.test_data_path, sep=self.delimiter, names=self.file_header, engine='python').loc[:,
                         ['user_id', 'item_id', 'rating']]

        self.num_users = len(set(self.test_data.user_id.unique()) | set(self.train_data.user_id.unique()))
        self.num_items = len(set(self.test_data.item_id.unique()) | set(self.train_data.item_id.unique()))

        if self.enable_logging:
            print("Number of users:", self.num_users, ", Number of items:", self.num_items)
            print("Train set size:", self.train_data.shape[0], ", Test set size:", self.test_data.shape[0])

    def _convert_to_matrix(self, dataframe):
        """
        Converts a pandas dataframe to a sparse matrix.
        将pandas数据框转换为稀疏矩阵。
        """
        rows, cols, scores, binary_scores = [], [], [], []
        for entry in dataframe.itertuples():
            user_id, item_id, rating = list(entry)[1:]
            binary_rating = 1 if rating >= self.like_threshold else 0

            rows.append(user_id)
            cols.append(item_id)
            scores.append(rating)
            binary_scores.append(binary_rating)

        score_matrix = csr_matrix((scores, (rows, cols)), shape=(self.num_users, self.num_items))
        binary_score_matrix = csr_matrix((binary_scores, (rows, cols)), shape=(self.num_users, self.num_items))
        return score_matrix, binary_score_matrix

    def calculate_global_mean_std(self):
        """
        Calculates global mean and standard deviation of the train matrix.
        计算训练矩阵的全局均值和标准差。
        """
        return self.train_matrix.data.mean(), self.train_matrix.data.std()

    def calculate_all_mean_std(self):
        """
        Calculates global and item-specific mean and standard deviation.
        计算全局和特定物品的均值和标准差。
        """
        all_calculated = True
        for attr in ['global_mean', 'global_std', 'item_means', 'item_stds']:
            if not hasattr(self, attr):
                all_calculated = False
                break
        if not all_calculated:
            global_mean, global_std = self.calculate_global_mean_std()
            item_means, item_stds = [global_mean] * self.num_items, [global_std] * self.num_items
            transposed_train_matrix = self.train_matrix.transpose()
            for item_id in range(self.num_items):
                item_vector = transposed_train_matrix.getrow(item_id).toarray()[0]
                item_ratings = item_vector[np.nonzero(item_vector)]
                if len(item_ratings) > 0:
                    item_means[item_id], item_stds[item_id] = item_ratings.mean(), item_ratings.std()
            self.global_mean, self.global_std, self.item_means, self.item_stds = global_mean, global_std, item_means, item_stds
        return self.global_mean, self.global_std, self.item_means, self.item_stds

    def calculate_item_popularity(self):
        """
        Calculates the popularity of each item based on the number of ratings.
        根据评分数量计算每个物品的流行度。
        """
        item_popularity = dict(self.train_data.groupby('item_id').size())
        item_pops = [0] * self.num_items
        for item_id in item_popularity.keys():
            item_pops[item_id] = item_popularity[item_id]
        return item_pops

    def find_user_unrated_items(self):
        """
        Finds items that a user has not rated.
        查找用户未评分的物品。
        """
        unrated_indicator = self.train_matrix.toarray()
        unrated_indicator[unrated_indicator > 0] = 1
        unrated_indicator = 1 - unrated_indicator
        user_unrated_items = {}
        for user_id in range(self.num_users):
            user_unrated_items[user_id] = list(unrated_indicator[user_id].nonzero()[0])
        return user_unrated_items

    def find_item_unrated_users(self, item_id):
        """
        Finds users who have not rated a specific item.
        查找没有给特定物品评分的用户。
        """
        item_vector = np.squeeze(self.train_matrix[:, item_id].toarray())
        item_vector[item_vector > 0] = 1
        unrated_indicator = 1 - item_vector
        return list(unrated_indicator.nonzero()[0])
