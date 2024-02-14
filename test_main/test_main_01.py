from AUSH.utils.load_data.load_attack_info import load_attack_info
from AUSH.utils.load_data.load_data import load_data


def info_test(data_dir_num,data_set_name, target_id):
    base_dir = "/Users/wangkai/PycharmProjects/ShillingAttack/AUSH/"
    path_train = base_dir + 'data/data/' + str(data_dir_num) + "/" + data_set_name + '_train.dat'
    path_test = base_dir + 'data/data/' + str(data_dir_num) + "/" + data_set_name + '_test.dat'
    attack_info_path = [base_dir + "data/data/" + str(data_dir_num) + "/" + data_set_name + "_selected_items",
                        base_dir + "data/data/" + str(data_dir_num) + "/" + data_set_name + "_target_users"]

    dataset_class = load_data(path_train=path_train, path_test=path_test, header=['user_id', 'item_id', 'rating'],
                              sep='\t', print_log=True)

    attack_info = load_attack_info(*attack_info_path)

    selected_items = attack_info[target_id][0]

    print(selected_items)

if __name__ == '__main__':
    info_test(3,"filmTrust",1884)