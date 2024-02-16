import sys

sys.path.append("../../")
from NeuralShillMatrix.utils.evaluation_utils import rec_trainer
from NeuralShillMatrix.utils.data_loader import *
from NeuralShillMatrix.utils.tool_aids import *

data_subdir = '../data/raw_data'
data_attacked_subdir = '../data/attack_data'
model_subdir = '../result/model_checkpoint'
result_subdir = '../result/model_evaluation'
def train_rec(data_set_name, model_name, attack_method, target_id, is_train,  target_dir=None):

    if target_dir == None:
        if attack_method == "no":
            model_path = os.path.join(model_subdir, '_'.join([model_name, data_set_name]) + ".ckpt")
            train_data_attacked_path = os.path.join(data_subdir, f'{data_set_name}_train.dat')
        else:
            model_path = os.path.join(model_subdir, '_'.join([model_name, data_set_name, attack_method]) + ".ckpt")
            train_data_attacked_path = os.path.join(data_attacked_subdir,'_'.join([data_set_name, str(target_id), attack_method]) + ".dat")

        dst_path = os.path.join(result_subdir, '_'.join([model_name, data_set_name, str(target_id), attack_method]))
    else:
        target_info = target_dir.split('/')
        new_model_path = f'../result/model_checkpoint/{target_info[3]}/{target_info[4]}'
        if not os.path.exists(new_model_path): os.makedirs(new_model_path)
        new_output_path = f'../result/model_evaluation/{target_info[3]}/{target_info[4]}'
        if not os.path.exists(new_output_path): os.makedirs(new_output_path)

        model_path = os.path.join(new_model_path, '_'.join([model_name, data_set_name, attack_method]) + ".ckpt")
        train_data_attacked_path = os.path.join(target_dir,'_'.join([data_set_name, str(target_id), attack_method]) + ".dat")
        dst_path = os.path.join(new_output_path, '_'.join([model_name, data_set_name, str(target_id), attack_method]))

    test_data_path = os.path.join(data_subdir,f'{data_set_name}_test.dat')
    dataset_class = DataLoader(path_to_train_data=train_data_attacked_path, path_to_test_data=test_data_path,
                              file_header=['user_id', 'item_id', 'rating'],
                              delimiter='\t', enable_logging=True)

    predictions, hit_ratios, target_ranks = rec_trainer(model_name, dataset_class, target_id, is_train,model_path)
    print(f"Completed evaluation: {attack_method} {dst_path}")
    write_predictions_to_file(predictions, hit_ratios, target_ranks, dst_path)

def run_entry(args, target_dir=None):

    for one_attack_method in args.attack_methods:
        if one_attack_method == 'no':
            attack_method_ = one_attack_method
        else:
            attack_method_ = '_'.join([one_attack_method, str(args.attack_count), str(args.filler_count)])

        is_train = 1
        train_rec(args.dataset_name, args.model_name, attack_method_, args.targets[0], is_train=is_train, target_dir=target_dir)

        for target in args.targets[1:]:
            if args.attack_method == 'no':
                is_train = 0
            train_rec(args.dataset, args.model_name, attack_method_, target, is_train=is_train, target_dir=target_dir)

if __name__ == '__main__':

    args = parse_arguments()

    if args.multiple_objectives == 1:

        for target in args.targets:
            target_item_path = '../data/merged_data' + f"/{target}"
            contents_nums = os.listdir(target_item_path)
            for num in contents_nums:
                if num != '1':
                    target_directory = target_item_path + f"/{num}"
                    run_entry(args,target_dir=target_directory)
    else:
        run_entry(args)






