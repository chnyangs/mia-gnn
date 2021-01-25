import torch
import torch.nn as nn
from sklearn.metrics import classification_report,precision_recall_fscore_support
import numpy as np
from torch.utils.data import DataLoader
import warnings

from attack_models import MLP
from utils import load_pickled_data, trainData, binary_acc, testData

warnings.simplefilter("ignore")


def apply_attack(epochs, attack_base_path, target_base_path):
    # Shadow Dataset Used as Attack Dataset
    S_X_train_in = load_pickled_data(attack_base_path + 'S_X_train_Label_1.pickle')
    S_y_train_in = load_pickled_data(attack_base_path + 'S_y_train_Label_1.pickle')
    S_X_train_out = load_pickled_data(attack_base_path + 'S_X_train_Label_0.pickle')
    S_y_train_out = load_pickled_data(attack_base_path + 'S_y_train_Label_0.pickle')

    # Target Dataset used as Attack Evaluation Dataset
    T_X_train_in = load_pickled_data(target_base_path + 'T_X_train_Label_1.pickle')
    T_y_train_in = load_pickled_data(target_base_path + 'T_y_train_Label_1.pickle')
    T_X_train_out = load_pickled_data(target_base_path + 'T_X_train_Label_0.pickle')
    T_y_train_out = load_pickled_data(target_base_path + 'T_y_train_Label_0.pickle')

    # Prepare Dataset
    X_attack = torch.FloatTensor(np.concatenate((S_X_train_in, S_X_train_out), axis=0))
    y_target = torch.FloatTensor(np.concatenate((T_y_train_in, T_y_train_out), axis=0))
    y_attack = torch.FloatTensor(np.concatenate((S_y_train_in, S_y_train_out), axis=0))
    X_target = torch.FloatTensor(np.concatenate((T_X_train_in, T_X_train_out), axis=0))

    n_in_size = X_attack.shape[1]
    # Init Attack Model
    attack_model = MLP(in_size=n_in_size, out_size=1, hidden_1=64, hidden_2=64)
    attack_criterion = nn.BCEWithLogitsLoss()
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.0001)
    attack_train_data = trainData(X_attack, y_attack)
    # Prepare Attack Model Training Data
    attack_train_loader = DataLoader(dataset=attack_train_data, batch_size=64, shuffle=True)
    for i in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in attack_train_loader:
            attack_optimizer.zero_grad()
            y_pred = attack_model(X_batch)
            loss = attack_criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            attack_optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print(f'Epoch {i + 0:03}: | Loss: {epoch_loss / len(attack_train_loader):.5f} |'
              f' Acc: {epoch_acc / len(attack_train_loader):.3f}')

    # Load Target Evaluation Data
    y_pred_list = []
    target_evaluate_data = testData(X_target)
    target_evaluate_loader = DataLoader(dataset=target_evaluate_data, batch_size=1)
    # Eval Attack Model
    attack_model.eval()
    print("Attack Start......")
    with torch.no_grad():
        for X_batch in target_evaluate_loader:
            y_test_pred = attack_model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    report = classification_report(y_target, y_pred_list)
    precision, recall, fscore, support = precision_recall_fscore_support(y_target,y_pred_list, average='macro')
    print("Attack Precision:{}, Recall:{} and F-Score:{}".format(precision, recall, fscore))


if __name__ == '__main__':
    attack_path = 'out/superpixels_graph_classification/checkpoints/GCN_CIFAR10_GPU1_20h54m19s_on_Jan_25_2021/S_RUN_/'
    target_path = 'out/superpixels_graph_classification/checkpoints/GCN_CIFAR10_GPU1_20h54m19s_on_Jan_25_2021/T_RUN_/'
    apply_attack(300,attack_base_path=attack_path, target_base_path=target_path)
