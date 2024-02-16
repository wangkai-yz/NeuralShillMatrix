import argparse
import os
import shutil

def parse_attack_info(selected_items_path, target_users_path):
    """
    Parse the attack information from the provided files.
    从提供的文件中解析攻击信息。

    Args:
        selected_items_path (str): Path to the file containing selected items information.
        target_users_path (str): Path to the file containing target users information.

    Returns:
        dict: A dictionary containing attack information.
    """
    attack_info = {}
    with open(selected_items_path, "r") as fin:
        for line in fin:
            line = line.strip("\n").split("\t")
            target_item, selected_items = int(line[0]), list(map(int, line[1].split(",")))
            attack_info[target_item] = [selected_items]
    with open(target_users_path, "r") as fin:
        for line in fin:
            line = line.strip("\n").split("\t")
            target_item, target_users = int(line[0]), list(map(int, line[1].split(",")))
            attack_info[target_item].append(target_users)

    return attack_info

def append_attack_data_to_file(original_file_path, modified_file_path, synthetic_profiles, original_user_count):
    """
    将生成的假用户配置文件追加到原始数据文件中，创建一个包含攻击数据的新文件。

    Args:
        original_file_path: 原始数据文件的路径。
        modified_file_path: 修改后，包含攻击数据的文件的路径。
        synthetic_profiles: 生成的假用户评分矩阵列表。
        original_user_count: 原始数据中的用户数量。
    """
    attack_data_lines = []
    for index, profile in enumerate(synthetic_profiles):
        # 获取该假用户评分的项目ID和评分值
        rated_item_indices = profile.nonzero()[0]
        ratings = profile[rated_item_indices]
        # 为每个评分构建字符串，并增加用户索引偏移
        profile_lines = [
            f"{original_user_count + index}\t{item_id}\t{rating}"
            for item_id, rating in zip(rated_item_indices, ratings)
        ]
        attack_data_lines.append('\n'.join(profile_lines) + '\n')

    # 如果目标文件已存在，先删除
    if os.path.exists(modified_file_path):
        os.remove(modified_file_path)
    # 复制原始文件到新位置
    shutil.copyfile(original_file_path, modified_file_path)
    # 追加攻击数据
    with open(modified_file_path, 'a+') as file_out:
        file_out.writelines(attack_data_lines)

def write_predictions_to_file(predicted_scores, user_hit_ratios, item_rankings, output_file_path):
    """
    将预测评分、命中率和目标项目排名写入指定的文件。

    Args:
        predicted_scores: 预测评分的列表，列表中的每个元素对应一个用户的预测评分。
        user_hit_ratios: 每个用户对应的命中率列表，列表的每个元素是一个包含不同Top N命中率的列表。
        item_rankings: 目标项目在每个用户推荐列表中的排名，如果目标项目不在Top N中，则为None。
        output_file_path: 输出文件的路径。

    此函数不返回任何值，但会在指定的路径创建或覆盖一个文件，文件中包含了用户ID、预测评分、命中率和目标项目排名的信息。
    """
    compiled_data = []  # 初始化一个空列表来收集所有要写入文件的数据行
    for user_id, score in enumerate(predicted_scores):
        # 获取用户的目标项目排名，如果未在Top N中则显示"Not in Top N"
        rank_message = item_rankings.get(user_id, "Not in Top N")
        # 构建要写入文件的字符串，包括用户ID、预测评分、命中率和排名
        data_line = '\t'.join(map(str, [user_id, score] + user_hit_ratios[user_id] + [rank_message]))
        compiled_data.append(data_line)

    # 将所有数据行写入指定的文件路径
    with open(output_file_path, 'w') as file_out:
        file_out.write('\n'.join(compiled_data))

def parse_arguments():
    """
    Parse command line arguments.
    解析命令行参数。

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default='filmTrust', help='filmTrust/ml100k/grocery')

    parser.add_argument('--attack_methods', type=str, default='gan,BigGan,average,segment,random,bandwagon', help='gan,average,segment,random,bandwagon')

    parser.add_argument('--model_name', type=str, default='NNMF', help='NNMF,IAutoRec,UAutoRec,NMF_25')

    parser.add_argument('--targets', type=str, default='1689',help='Target item id')

    parser.add_argument('--attack_count', type=int, default=50, help='fixed 50')

    parser.add_argument('--filler_count', type=int, default=16, help='90 for ml100k,36 for filmTrust')

    parser.add_argument('--filler_method', type=str, default='', help='0/1/2/3')

    parser.add_argument('--bandwagon_selected', type=str, default='103,98,115',help='The target item of the trend selection')

    parser.add_argument('--multiple_objectives', type=int, default='1',help='0&1')

    args = parser.parse_args()
    #
    args.attack_methods = args.attack_methods.split(',')
    args.targets = list(map(int, args.targets.split(',')))
    args.bandwagon_selected = list(map(int, args.bandwagon_selected.split(',')))

    return args
