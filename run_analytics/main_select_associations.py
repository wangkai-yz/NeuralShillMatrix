import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from scipy.spatial.distance import cosine

# def cosine_similarity_calculation(data):
#     """
#     计算用户间的余弦相似性
#     """
#     # 创建用户和项目的矩阵
#     rating_matrix = data.pivot_table(index='uid', columns='iid', values='rating').fillna(0)
#
#     # 计算用户间的余弦相似性
#     user_similarity = pd.DataFrame(cosine_similarity(rating_matrix), index=rating_matrix.index,
#                                    columns=rating_matrix.index)
#
#     # 创建网络图
#     G = nx.Graph()
#
#     # Add nodes (users)
#     for user in rating_matrix.index:
#         G.add_node(user)
#
#     # 根据相似度在用户之间添加边
#     # 仅为相似度高于一定阈值的用户添加边，以避免生成完整的图
#     threshold = 0.5  # This threshold can be adjusted
#     for user1 in rating_matrix.index:
#         for user2 in rating_matrix.index:
#             if user1 != user2 and user_similarity.loc[user1, user2] > threshold:
#                 G.add_edge(user1, user2, weight=user_similarity.loc[user1, user2])
#
#     return G.number_of_nodes(), G.number_of_edges()

def item_similarity_calculation(data):
    # 创建用户和项目的矩阵
    rating_matrix = data.pivot_table(index='iid', columns='uid', values='rating').fillna(0)

    # 计算项目间的余弦相似性
    item_similarity = pd.DataFrame(cosine_similarity(rating_matrix), index=rating_matrix.index,
                                   columns=rating_matrix.index)
    return item_similarity

def find_least_related_items(data, num_items=3):
    item_similarity = item_similarity_calculation(data)

    # 计算每个项目与所有其他项目的平均相似度
    average_similarity = item_similarity.mean()

    # 找出平均相似度最低的项目
    least_related_items = average_similarity.nsmallest(num_items).index
    return least_related_items

def cosine_similarity_calculation(data):
    """
    计算项目间的余弦相似性
    """
    # 创建用户和项目的矩阵，并进行转置以得到项目-用户矩阵
    item_matrix = data.pivot_table(index='iid', columns='uid', values='rating').fillna(0)

    # 计算项目间的余弦相似性
    item_similarity = pd.DataFrame(cosine_similarity(item_matrix), index=item_matrix.index,
                                   columns=item_matrix.index)
    return item_similarity

def find_least_similar_items(item_similarity, target_items):
    """
    在目标项目列表中找出最不相关的三个项目。
    """
    min_similarity = {}
    for item in target_items:
        other_items = [i for i in target_items if i != item]
        min_similarity[item] = min(item_similarity.loc[item, other_items])

    # 根据最小相似度排序并取前三个
    least_similar_items = sorted(min_similarity, key=min_similarity.get)[:3]
    return least_similar_items

def find_most_similar_triplet(item_similarity, target_items):
    """
    在目标项目列表中找出三个彼此之间最相关的项目。
    """
    best_triplet = None
    highest_avg_similarity = -np.inf  # 初始化为负无穷大

    # 遍历所有可能的三项目组合
    for triplet in combinations(target_items, 3):
        # 计算组合内项目的平均相似度
        avg_similarity = item_similarity.loc[triplet, triplet].sum().sum() / 6  # 每组有3个项目，因此有6对不同的组合

        # 更新最高平均相似度的组合
        if avg_similarity > highest_avg_similarity:
            highest_avg_similarity = avg_similarity
            best_triplet = triplet

    return best_triplet

def get_target_items_similarity(item_similarity, target_items):
    """
    输出目标项目列表的余弦相似性。
    """
    # 选取目标项目之间的相似度
    target_similarity = item_similarity.loc[target_items, target_items]
    return target_similarity

if __name__ == '__main__':
    """
    样例数据：
    uid	    iid	    rating
    1079	245	    4.0
    870	    83	    0.5
    164	    803	    1.5
    1467	397	    4.0
    1224	216	    2.5
    545	    899	    4.0
    1063	16	    3.0
    1500	235	    2.0
    948	    3	    3.5
    617	    1	    3.0
    949	    211	    4.0
    567	    2	    0.5
    """
    # 加载数据
    data = pd.read_csv('/AUSH/data/data/ratings.txt',
                       sep=' ', header=None, names=['uid', 'iid', 'rating'])

    # nodes,edges= cosine_similarity_calculation(data)
    # print(nodes,edges)

    item_similarity = cosine_similarity_calculation(data)
    # target_items = [920, 1101, 393, 262, 1898, 1897, 1744, 1745, 356, 259, 1103, 1104, 1366, 1369, 719, 721, 1175, 1543, 54, 29]  # 示例目标项目ID列表
    # least_similar_items = find_least_similar_items(item_similarity, target_items)
    # print("最不相关的三个目标项目ID:", least_similar_items)

    # target_items_most_similar = find_most_similar_triplet(item_similarity, target_items)
    # print("每个目标项目与之最相关的三个其他项目:\n", target_items_most_similar)

    # target_items = [529, 1123, 1282, 746, 1853, 1884, 1734, 1735, 2042, 1917, 1344, 36, 1447, 78, 824, 825, 501, 748, 313, 422]  # 示例目标项目ID列表
    target_items = [1898, 1897, 1369]
    target_items_similarity = get_target_items_similarity(item_similarity, target_items)
    print("目标项目列表之间的余弦相似性:\n", target_items_similarity)


