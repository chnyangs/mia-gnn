import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings

from attack_models import MLP
from utils import load_pickled_data, select_top_k, binary_acc, testData, trainData

warnings.simplefilter("ignore")


def transfer_based_attack(epochs):

    base_path = 'out/PROTEINS_full/GatedGCN_PROTEINS_full_GPU0_00h44m43s_on_Jan_03_2021/'
    base_path_test = 'out/OGBG/OGBG_PPA_100_57/'

    X_train_in = load_pickled_data(base_path + 'X_train_Label_1.pickle')
    y_train_in = load_pickled_data(base_path + 'y_train_Label_1.pickle')
    X_train_out = load_pickled_data(base_path + 'X_train_Label_0.pickle')
    y_train_out = load_pickled_data(base_path + 'y_train_Label_0.pickle')
    feature_nums = 2
    X_train_in = select_top_k(X_train_in, feature_nums)
    X_train_out = select_top_k(X_train_out, feature_nums)
    split_index = min(len(X_train_in), len(X_train_out))
    index_list = list(range(0, split_index))
    random.shuffle(index_list)
    X = np.concatenate((X_train_in[index_list], X_train_out[index_list]), axis=0)
    y = np.concatenate((y_train_in[index_list], y_train_out[index_list]), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=111)
    dataset = (torch.FloatTensor(x_train), torch.FloatTensor(y_train),
               torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape[1]
    model = MLP(in_size=n_in, out_size=1, hidden_1=64, hidden_2=64)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # train_loader = DataLoader(train_y, batch_size=32, shuffle=True)
    train_data = trainData(train_x, train_y)
    test_data = testData(test_x)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    for i in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        # print(
        #     f'Epoch {i + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    report = classification_report(test_y, y_pred_list)
    precision, recall, fscore, support = precision_recall_fscore_support(test_y,
                                                                         y_pred_list, average='macro')
    # print('classification_report:', report)
    print(precision, recall)

    # load test dataset
    X_train_in_test = load_pickled_data(base_path_test + 'X_train_Label_1.pickle')
    y_train_in_test = load_pickled_data(base_path_test + 'y_train_Label_1.pickle')
    X_train_out_test = load_pickled_data(base_path_test + 'X_train_Label_0.pickle')
    y_train_out_test = load_pickled_data(base_path_test + 'y_train_Label_0.pickle')

    # feature_nums = 10
    X_train_in_test = select_top_k(X_train_in_test, feature_nums)
    X_train_out_test = select_top_k(X_train_out_test, feature_nums)
    split_index_test = min(len(X_train_in_test), len(X_train_out_test))
    index_list_test = list(range(0, split_index_test))
    random.shuffle(index_list_test)
    X_t = np.concatenate((X_train_in_test[index_list_test], X_train_out_test[index_list_test]), axis=0)
    y_t = np.concatenate((y_train_in_test[index_list_test], y_train_out_test[index_list_test]), axis=0)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(X_t, y_t, test_size=0.2,
                                                            shuffle=True)

    test_data = testData(torch.FloatTensor(X_t))
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    report = classification_report(torch.FloatTensor(y_t), y_pred_list)
    precision, recall, fscore, support = precision_recall_fscore_support(torch.FloatTensor(y_t),
                                                                         y_pred_list, average='macro')
    # print('classification_report:', report)
    # print('Transferred Attack:',precision, recall)
    print(precision, recall)
