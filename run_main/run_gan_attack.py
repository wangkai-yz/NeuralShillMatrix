import sys
import numpy as np
sys.path.append("../../")
import os, argparse
from NeuralShillMatrix.utils.data_loader import *
from NeuralShillMatrix.utils.tool_aids import *
from NeuralShillMatrix.model.train_attacker import Train_Attacker

data_subdir = '..\data\\raw_data'
data_attack_subdir = '..\data\\attack_data'
model_subdir = '..\\result\model_checkpoint'
def gan_attack(data_set_name, attack_method, target_id, final_attack_setting=None):

    train_data_path = os.path.join(data_subdir, f'{data_set_name}_train.dat')
    test_data_path = os.path.join(data_subdir,f'{data_set_name}_test.dat')
    attack_info_paths = [
        os.path.join(data_subdir, f"{data_set_name}_selected_items"),
        os.path.join(data_subdir, f"{data_set_name}_target_users")
    ]

    model_path = os.path.join(model_subdir, '_'.join([data_set_name, attack_method, str(target_id)]) + ".ckpt")



    attack_info = parse_attack_info(*attack_info_paths)
    dataset_class = DataLoader(path_to_train_data=train_data_path, path_to_test_data=test_data_path, file_header=['user_id', 'item_id', 'rating'],
                              delimiter='\t', enable_logging=True)

    if len(attack_method.split('_')[1:]) == 2:
        attack_num, filler_num = map(int, attack_method.split('_')[1:])
        filler_method = 0
    else:
        attack_num, filler_num, filler_method = map(int, attack_method.split('_')[1:])
    selected_items = attack_info[target_id][0]

    #
    gan_attacker = Train_Attacker(dataset_class, params_D=None, params_G=None, target_id=target_id,
                                      selected_id_list=selected_items,
                                      filler_num=filler_num, attack_num=attack_num, filler_method=filler_method)

    fake_profiles, real_profiles, filler_indicator = gan_attacker.execute(is_train=1, model_path=model_path,
                                                                          final_attack_setting=final_attack_setting)
    gan_attacker.sess.close()

    dst_path = os.path.join(data_attack_subdir,'_'.join([args.dataset_name, str(target_id), attack_method]) + ".dat")
    append_attack_data_to_file(train_data_path, dst_path, fake_profiles, dataset_class.num_users)


if __name__ == '__main__':

    args = parse_arguments()

    for one_attack_method in args.attack_methods:

        attack_method = '_'.join([one_attack_method, str(args.attack_count), str(args.filler_count), str(args.filler_method)]).strip('_')

        for target_id in args.targets:
            attackSetting_path = '_'.join(map(str, [args.dataset_name, args.attack_count, args.filler_count, target_id]))
            attackSetting_path = os.path.join(data_attack_subdir, attackSetting_path + '_attackSetting')

            real_profiles, filler_indicator = np.load(attackSetting_path + '.npy')
            final_attack_setting = [args.attack_count, real_profiles, filler_indicator]
            gan_attack(args.dataset_name, attack_method, target_id, final_attack_setting=final_attack_setting)
