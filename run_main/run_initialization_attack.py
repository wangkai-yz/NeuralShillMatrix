import sys
import os

sys.path.append("../../")
from NeuralShillMatrix.utils.data_loader import *
from NeuralShillMatrix.utils.tool_aids import *
from NeuralShillMatrix.model.train_attacker import Train_Attacker

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

if __name__ == '__main__':

    args = parse_arguments()

    data_set, attack_info = load_dataset_attack_info(args.dataset_name)

    full_path = os.path.join('../data', 'attack_data')
    os.makedirs(full_path, exist_ok=True)

    for target_id in args.targets:

        attackSetting_path = '_'.join(map(str, [args.dataset_name, args.attack_count, args.filler_count, target_id]))
        attackSetting_path = os.path.join(full_path, attackSetting_path + '_attackSetting')

        gan_attacker = Train_Attacker(data_set, params_D=None, params_G=None,
                                          target_id=target_id,
                                          selected_id_list=attack_info[target_id][0],
                                          filler_num=args.filler_count,
                                          attack_num=args.attack_count,
                                          filler_method=0)
        _, real_profiles, filler_indicator = gan_attacker.execute(is_train=0, model_path='no',final_attack_setting=[args.attack_count, None, None])
        np.save(attackSetting_path, [real_profiles, filler_indicator])
