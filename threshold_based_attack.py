import os
import pickle
import torch
from scipy.spatial import distance
import numpy as np
from torch import nn



def load_pickled_data(path):
    with open(path, 'rb') as f:
        unPickler = pickle.load(f)
        return unPickler


def load_data(m_path, nm_path):
    data_in = load_pickled_data(m_path)
    data_out = load_pickled_data(nm_path)
    return data_in, data_out


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def cal_distances(data):
    # print('calculate distances...')
    distance_matrix = []
    for raw in data:
        label = np.argmax(raw)
        # cosine_dis = distance.cosine(label, raw)
        euclid_dis = distance.euclidean(label, raw)
        # corr_dis = distance.correlation(np.argmax(raw), raw) # nan
        cheby_dis = distance.chebyshev(label, raw)
        bray_dis = distance.braycurtis(label, raw)
        canber_dis = distance.canberra(label, raw)
        mahal_dis = distance.cityblock(label, raw)
        sqeuclid_dis = distance.sqeuclidean(label, raw)
        v = [euclid_dis, cheby_dis, bray_dis, canber_dis, mahal_dis, sqeuclid_dis]
        distance_matrix.append(v)
    return distance_matrix


def cal_distance(data):
    label = np.argmax(data)
    # cosine_dis = distance.cosine(label, raw)
    euclid_dis = distance.euclidean(label, data)
    # corr_dis = distance.correlation(np.argmax(raw), raw) # nan
    cheby_dis = distance.chebyshev(label, data)
    bray_dis = distance.braycurtis(label, data)
    canber_dis = distance.canberra(label, data)
    mahal_dis = distance.cityblock(label, data)
    sqeuclid_dis = distance.sqeuclidean(label, data)
    v = [euclid_dis, cheby_dis, bray_dis, canber_dis, mahal_dis, sqeuclid_dis]
    return v


# Setup a plot such that only the bottom spine is shown
def get_all_probabilities(data, factor):
    return_list = []
    for d in data:
        return_list.append(d[np.argmax(d)] * factor)
    return return_list


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def get_all_exps(path):
    dataset_list = ['CIFAR10', 'MNIST', 'DD', 'ENZYMES', 'PROTEINS_full', 'OGBG']
    folders = os.listdir(path)
    assert len(folders) > 0, "No dataset folder exist!"
    exp_path_list = []
    for folder in folders:
        if dataset_list.__contains__(folder):
            dataset_folder = os.path.join(path, folder)
            exps = os.listdir(dataset_folder)
            for exp in exps:
                exp_path = os.path.join(dataset_folder, exp)
                exp_path_list.append(exp_path)
    return exp_path_list


if __name__ == '__main__':
    folders = os.listdir('out/DD')
    # print(sorted(folders))
    for folder in folders:
        all_exps = os.listdir('out/DD/' + folder)
        for exp_path in all_exps:
            exp_path = os.path.join('out/DD/' + folder, exp_path)
            # print(exp_path)
            if os.listdir(exp_path).__contains__('S_RUN_'):
                exp_path = os.path.join(exp_path, 'S_RUN_')
                m_data_path = os.path.join(exp_path, 'S_X_train_Label_1.pickle')
                nm_data_path = os.path.join(exp_path, 'S_X_train_Label_0.pickle')
                data_in, data_out = load_data(m_data_path, nm_data_path)
                # LOSS function based attack
                # ce_criterion = nn.CrossEntropyLoss()
                # nl_criterion = nn.NLLLoss()
                mse_criterion = nn.MSELoss()
                ce_criterion = nn.CrossEntropyLoss()
                mse_in_loss_list, mse_out_loss_list, loss_diff_list = [], [], []
                ce_in_loss_list, ce_out_loss_list, ce_loss_diff_list = [], [], []
                with open('out/DD/dd_loss_difference_calculation_single_instance.txt', 'a+') as writer:
                    writer.write("For Experiment:{} \n".format(exp_path))
                    for i in range(min(len(data_in), len(data_out))):
                        x_in,x_in_label = data_in[i],np.argmax(data_in[i])
                        x_out,x_out_label = data_out[i],np.argmax(data_out[i])

                        ce_in_loss = ce_criterion(torch.FloatTensor([x_in]), torch.LongTensor([x_in_label]))
                        ce_out_loss = ce_criterion(torch.FloatTensor([x_out]), torch.LongTensor([x_out_label]))

                        mse_in_loss = mse_criterion(torch.FloatTensor([x_in]), torch.LongTensor([x_in_label]))
                        mse_out_loss = mse_criterion(torch.FloatTensor([x_out]), torch.LongTensor([x_out_label]))

                        mse_in_loss_list.append(float(mse_in_loss.numpy()))
                        mse_out_loss_list.append(float(mse_out_loss.numpy()))

                        ce_in_loss_list.append(float(ce_in_loss.numpy()))
                        ce_out_loss_list.append(float(ce_out_loss.numpy()))

                    loss_diff_list.append(np.mean(mse_in_loss_list) - np.mean(mse_out_loss_list))
                    # writer.write("MSE Difference:{}\n".format(loss_diff_list))
                    print(np.mean(ce_in_loss_list), np.mean(ce_out_loss_list))
                    writer.write(
                        "\t\tMSELLoss for Member:\n\t{} and Non-Member：\n\t{}\n".format(ce_in_loss_list,
                                                                                        ce_out_loss_list))
