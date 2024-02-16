import sys

sys.path.append("../../")
from NeuralShillMatrix.utils.data_loader import *
from NeuralShillMatrix.utils.tool_aids import *
from NeuralShillMatrix.model.attacker.baseline import BaselineAttack

data_subdir = '../data/raw_data'
def load_dataset_attack_info(dataset_name):
    """
    Load dataset and corresponding attack information.

    Returns:
        tuple: A tuple containing dataset class and attack information.
    """

    train_data_path = os.path.join(data_subdir, f'{dataset_name}_train.dat')
    test_data_path = os.path.join(data_subdir,f'{dataset_name}_test.dat')

    dataset = DataLoader(path_to_train_data=train_data_path,
                         path_to_test_data=test_data_path,
                         file_header=['user_id', 'item_id', 'rating'],
                         delimiter='\t', enable_logging=False)

    attack_info_paths = [
        os.path.join(data_subdir, f"{dataset_name}_selected_items"),
        os.path.join(data_subdir, f"{dataset_name}_target_users")
    ]

    loaded_attack_info = parse_attack_info(*attack_info_paths)

    return dataset, loaded_attack_info

def generate_baseline_attack_profiles(data_source, attack_config, attack_strategy, target_item_id, popular_items,
                                      specified_fillers=None):
    """
    根据基线攻击策略生成假用户评分配置文件。

    Args:
        data_source: 包含数据集信息的类实例。
        attack_config: 目标项目的攻击信息配置。
        attack_strategy: 指定的攻击策略，如 random, bandwagon, average, 或 segment。
        target_item_id: 目标项目的ID。
        popular_items: 用于bandwagon攻击的流行项目列表。
        specified_fillers: 指定的填充项指示器，可用于固定填充项的场景。

    Returns:
        生成的假用户评分配置文件矩阵。
    """
    selected_items, _ = attack_config[target_item_id]
    attack_type, num_fake_profiles, num_fillers = attack_strategy.split('_')
    num_fake_profiles, num_fillers = int(num_fake_profiles), int(num_fillers)

    # 获取数据集全局和项目级的平均评分及标准差
    global_avg, global_std, item_avgs, item_stds = data_source.calculate_all_mean_std()

    # 初始化基线攻击器实例
    attacker = BaselineAttack(num_fake_profiles, num_fillers, data_source.num_items, target_item_id,
                              global_avg, global_std, item_avgs, item_stds, max_rating=5.0, min_rating=1.0,
                              fixed_filler_indicator=specified_fillers)

    # 根据攻击类型选择对应的攻击方法
    if attack_type == "random":
        fake_profiles = attacker.random_attack()
    elif attack_type == "bandwagon":
        fake_profiles = attacker.bandwagon_attack(popular_items)
    elif attack_type == "average":
        fake_profiles = attacker.average_attack()
    elif attack_type == "segment":
        fake_profiles = attacker.segment_attack(selected_items)
    else:
        raise ValueError('Unsupported attack method specified.')

    return fake_profiles

if __name__ == '__main__':

    args = parse_arguments()

    data_set, attack_info = load_dataset_attack_info(args.dataset_name)

    full_path = os.path.join('../data', 'attack_data')
    os.makedirs(full_path, exist_ok=True)

    for target_id in args.targets:

        for attack_method in args.attack_methods:
            attack_model = '_'.join([attack_method, str(args.attack_count), str(args.filler_count)])
            fake_profiles = generate_baseline_attack_profiles(data_set, attack_info, attack_model, target_id,args.bandwagon_selected, None)

            ori_path = os.path.join(data_subdir, f'{args.dataset_name}_train.dat')
            dst_path = os.path.join('../data/attack_data', '_'.join([args.dataset_name, str(target_id), attack_model]) + ".dat")
            append_attack_data_to_file(ori_path, dst_path, fake_profiles, data_set.num_users)