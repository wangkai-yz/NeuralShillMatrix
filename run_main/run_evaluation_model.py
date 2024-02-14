import sys

sys.path.append("../../")
from NeuralShillMatrix.utils.evaluation_utils import rec_trainer
from NeuralShillMatrix.utils.data_loader import *
from NeuralShillMatrix.utils.tool_aids import *

data_subdir = '..\data\\raw_data'
data_attacked_subdir = '..\data\\attack_data'
model_subdir = '..\\result\model_checkpoint'
result_subdir = '..\\result\model_evaluation'
def train_rec(data_set_name, model_name, attack_method, target_id, is_train):

    if attack_method == "no":
        model_path = os.path.join(model_subdir, '_'.join([model_name, data_set_name]) + ".ckpt")
        train_data_attacked_path = os.path.join(data_subdir,f'{data_set_name}_train.dat')
    else:
        model_path = os.path.join(model_subdir, '_'.join([model_name, data_set_name, attack_method]) + ".ckpt")
        train_data_attacked_path = os.path.join(data_attacked_subdir,'_'.join([data_set_name, str(target_id), attack_method]) + ".dat")

    test_data_path = os.path.join(data_subdir,f'{data_set_name}_test.dat')

    dataset_class = DataLoader(path_to_train_data=train_data_attacked_path, path_to_test_data=test_data_path,
                              file_header=['user_id', 'item_id', 'rating'],
                              delimiter='\t', enable_logging=True)

    predictions, hit_ratios, target_ranks = rec_trainer(model_name, dataset_class, target_id, is_train,
                                                              model_path)
    dst_path = os.path.join(result_subdir, '_'.join([model_name, data_set_name, str(target_id), attack_method]))
    dst_path = dst_path.strip('_')
    write_predictions_to_file(predictions, hit_ratios, target_ranks, dst_path)

if __name__ == '__main__':

    args = parse_arguments()

    for one_attack_method in args.attack_methods:
        if one_attack_method == 'no':
            attack_method_ = one_attack_method
        else:
            attack_method_ = '_'.join([one_attack_method, str(args.attack_count), str(args.filler_count)])

        is_train = 1
        train_rec(args.dataset_name, args.model_name, attack_method_, args.targets[0], is_train=is_train)

        for target in args.targets[1:]:
            if args.attack_method == 'no':
                is_train = 0
            train_rec(args.dataset, args.model_name, attack_method_, target, is_train=is_train)


