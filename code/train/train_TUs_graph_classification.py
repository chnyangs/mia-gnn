"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from train.metrics import accuracy_TU as accuracy

"""
    For GCNs
"""


def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()

        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    train_posterior = []
    train_labels = []
    flag = []
    if type(epoch) is str:
        flag = epoch.split('|')
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)

            # Calculate Posteriors
            if len(flag) == 3:
                for posterior in F.softmax(batch_scores, dim=1).detach().cpu().numpy().tolist():
                    train_posterior.append(posterior)
                    train_labels.append(int(flag[0]))

            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        # Save Posteriors
        if len(flag) == 3:
            x_save_path = flag[2] + '/' + flag[1] + '_X_train_Label_' + str(flag[0]) + '.pickle'
            y_save_path = flag[2] + '/' + flag[1] + '_y_train_Label_' + str(flag[0]) + '.pickle'
            print("save_path:", x_save_path, y_save_path)
            pickle.dump(np.array(train_posterior), open(x_save_path, 'wb'))
            pickle.dump(np.array(train_labels), open(y_save_path, 'wb'))
    return epoch_test_loss, epoch_test_acc


"""
    For WL-GNNs
"""


def train_epoch_dense(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)

        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels)
        loss.backward()

        if not (iter % batch_size):
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(scores, labels)
        nb_data += labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_dense(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)

            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(scores, labels)
            nb_data += labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data

    return epoch_test_loss, epoch_test_acc


def check_patience(all_losses, best_loss, best_epoch, curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter
