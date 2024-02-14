import numpy as np
import math

class BaselineAttack:
    def __init__(self, attack_count, filler_count, num_items, target_item_id,
                 global_mean_rating, global_rating_std, item_means, item_stds, max_rating, min_rating, fixed_filler_indicator=None):
        """
        Initialize BaselineAttack object with attack parameters and dataset statistics.
        使用攻击参数和数据集统计信息初始化BaselineAttack对象。

        Args:
            attack_count (int): The number of fake data users to generate.要生成的虚假数据用户的数量。
            filler_count (int): The number of additional fill items selected.额外选取的填充项目数量。
            num_items (int): Total number of items in the dataset.数据集中项目的总数。
            target_item_id (int): ID of the target item.目标项的ID
            global_mean_rating (float): Global mean rating of the dataset.数据集的全局平均评分。
            global_rating_std (float): Global standard deviation of ratings in the dataset.数据集中评级的全局标准差。
            item_means (numpy.ndarray): Mean ratings for each item.每个项目的平均评分。
            item_stds (numpy.ndarray): Standard deviations of ratings for each item.每个项目评级的标准偏差。
            max_rating (float): Maximum allowed rating.最大允许额定值。
            min_rating (float): Minimum allowed rating.允许的最小评级。
            fixed_filler_indicator (numpy.ndarray or None): Indicator for fixed fillers in each attack profile.每个攻击配置文件中固定填充物的指标。
        """
        self.attack_count = attack_count
        self.filler_count = filler_count
        self.num_items = num_items
        self.target_item_id = target_item_id
        self.global_mean_rating = global_mean_rating
        self.global_rating_std = global_rating_std
        self.item_means = item_means
        self.item_stds = item_stds
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.fixed_filler_indicator = fixed_filler_indicator

    def random_attack(self):
        """
        Generate random attack profiles.


        """
        filler_candidates = list(set(range(self.num_items)) - {self.target_item_id})
        fake_profiles = np.zeros(shape=[self.attack_count, self.num_items], dtype=float)

        fake_profiles[:, self.target_item_id] = self.max_rating

        for i in range(self.attack_count):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candidates, size=self.filler_count, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = np.random.normal(loc=self.global_mean_rating, scale=self.global_rating_std, size=self.filler_count)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.max_rating, r))
        return fake_profiles

    def bandwagon_attack(self, selected_item_ids):
        """
        Generate bandwagon attack profiles.

        Args:
            selected_item_ids (list): List of IDs of selected items.

        Returns:
            numpy.ndarray: Array of attack profiles.
        """
        filler_candidates = list(set(range(self.num_items)) - set([self.target_item_id] + selected_item_ids))
        fake_profiles = np.zeros(shape=[self.attack_count, self.num_items], dtype=float)

        fake_profiles[:, [self.target_item_id] + selected_item_ids] = self.max_rating

        for i in range(self.attack_count):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candidates, size=self.filler_count, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = np.random.normal(loc=self.global_mean_rating, scale=self.global_rating_std, size=self.filler_count)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.max_rating, r))
        return fake_profiles

    def average_attack(self):
        """
        Generate average attack profiles.

        Returns:
            numpy.ndarray: Array of attack profiles.
        """
        filler_candidates = list(set(range(self.num_items)) - {self.target_item_id})
        fake_profiles = np.zeros(shape=[self.attack_count, self.num_items], dtype=float)

        fake_profiles[:, self.target_item_id] = self.max_rating

        for i in range(self.attack_count):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candidates, size=self.filler_count, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            ratings = map(lambda iid: np.random.normal(loc=self.item_means[iid], scale=self.item_stds[iid], size=1)[0], fillers)
            for f_id, r in zip(fillers, ratings):
                fake_profiles[i][f_id] = max(math.exp(-5), min(self.max_rating, r))
        return fake_profiles

    def segment_attack(self, selected_item_ids):
        """
        Generate segment attack profiles.

        Args:
            selected_item_ids (list): List of IDs of selected items.

        Returns:
            numpy.ndarray: Array of attack profiles.
        """
        filler_candidates = list(set(range(self.num_items)) - set([self.target_item_id] + selected_item_ids))
        fake_profiles = np.zeros(shape=[self.attack_count, self.num_items], dtype=float)

        fake_profiles[:, [self.target_item_id] + selected_item_ids] = self.max_rating

        for i in range(self.attack_count):
            if self.fixed_filler_indicator is None:
                fillers = np.random.choice(filler_candidates, size=self.filler_count, replace=False)
            else:
                fillers = np.where(np.array(self.fixed_filler_indicator[i]) == 1)[0]
            fake_profiles[i][fillers] = self.min_rating
        return fake_profiles

