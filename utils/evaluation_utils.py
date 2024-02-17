try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from NeuralShillMatrix.model.IAutoRec import IAutoRec
from NeuralShillMatrix.model.NNMF import NNMF

def initialize_model_network(sess, model_name, dataset_class):
    """
    初始化指定的模型网络。
    Initialize the specified model network.

    参数 Parameters:
    - sess: TensorFlow会话。
    - model_name: 模型名称。
    - dataset_class: 数据集类实例。

    返回 Return:
    - model: 初始化的模型实例。
    """
    model = None
    if model_name == "IAutoRec":
        model = IAutoRec(sess, dataset_class)
    elif model_name == "NNMF":
        model = NNMF(sess, dataset_class)
    return model

def get_top_n_items_with_rank(model, n, target_id):
    """
    获取每个用户的top N推荐项及目标项的排名。
    Get top N recommended items for each user and the rank of the target item.

    参数 Parameters:
    - model: 模型实例。
    - n: 推荐列表的长度。
    - target_id: 目标项目ID。

    返回 Return:
    - top_n: 每个用户的top N推荐项。
    - target_ranks: 目标项在每个用户推荐列表中的排名。
    """
    top_n = {}
    target_ranks = {}
    user_nonrated_items = model.dataset_class.find_user_unrated_items()

    for uid in range(model.num_user):
        items = user_nonrated_items[uid]
        ratings = model.predict([uid] * len(items), items)
        item_rating = list(zip(items, ratings))
        item_rating.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [x[0] for x in item_rating[:n]]
        if target_id in top_n[uid]:
            target_ranks[uid] = top_n[uid].index(target_id) + 1  # 排名从1开始
        else:
            target_ranks[uid] = None  # 表示目标项不在top N中

    return top_n, target_ranks


def predict_for_target_item(model, target_id):
    """
    针对目标项目进行预测。
    Make predictions for the target item.

    参数 Parameters:
    - model: 模型实例。
    - target_id: 目标项目ID。

    返回 Return:
    - target_predictions: 针对目标项目的预测。
    - hit_ratios: 命中率。
    - target_ranks: 目标项目的排名。
    """
    target_predictions = model.predict(list(range(model.num_user)), [target_id] * model.num_user)

    top_n, target_ranks = get_top_n_items_with_rank(model, 50, target_id)
    hit_ratios = {}
    for uid in top_n:
        hit_ratios[uid] = [1 if target_id in top_n[uid][:i] else 0 for i in [1, 3, 5, 10, 20, 50]]
    return target_predictions, hit_ratios, target_ranks

def train_evaluate_recommendation_model(model_name, dataset_class, target_id, is_train, model_path):
    """
    训练并评估推荐模型。
    Train and evaluate the recommendation model.

    参数 Parameters:
    - model_name: 模型名称。
    - dataset_class: 数据集类实例。
    - target_id: 目标项目ID。
    - is_train: 是否为训练模式。
    - model_path: 模型保存路径。

    返回 Return:
    - predictions: 预测结果。
    - hit_ratios: 命中率。
    - target_ranks: 目标项目的排名。
    """
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:

        rec_model = initialize_model_network(sess, model_name, dataset_class)
        if is_train:
            print('--> start train recommendation model...')
            rec_model.execute()
            rec_model.save(model_path)
        else:
            rec_model.restore(model_path)
        print('--> Start prediction for each user...')
        predictions, hit_ratios, target_ranks = predict_for_target_item(rec_model, target_id)
    return predictions, hit_ratios, target_ranks

