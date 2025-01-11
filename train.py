import os
import torch
import numpy as np
from tqdm import tqdm
from utils import load_data
from model import Model

to_generate_model_pkl_file = True
trainset = "/train_fasta"
trainlabel = "/train_label"
testset = "/test_fasta"
testlabel = "/test_label"

# path
Dataset_Path = './date/'
Result_Path = './result/'
Prot_Path ='./date/'

# Seed
SEED = 4396
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# Model parameters
NUMBER_EPOCHS = 1000
BATCH_SIZE = 256


def getlabel(fl):
    if fl >= 0.5:
        return 1
    else:
        return 0



def evaluate(model, val_features, val_graphs, val_labels):
    model.eval()
    epoch_loss_valid = 0.0
    exact_match = 0
    for i in tqdm(range(len(val_labels))):
        with torch.no_grad():

            sequence_features = torch.from_numpy(val_features[i])
            sequence_graphs = torch.from_numpy(val_graphs[i])

            labels = torch.from_numpy(np.array([int(float(val_labels[i]))]))

            sequence_features = torch.squeeze(sequence_features)
            sequence_graphs = torch.squeeze(sequence_graphs)


            if torch.cuda.is_available():
                features = sequence_features.cuda()
                graphs = sequence_graphs.cuda()

                y_true = labels.cuda().float()

            y_pred = model(features, graphs)
            new_y_pred = getlabel(y_pred)
            if (new_y_pred == y_true):
                exact_match += 1
            loss = model.criterion(y_pred, y_true)
            epoch_loss_valid += loss.item()
    epoch_loss_valid_avg = epoch_loss_valid / len(val_labels)
    acc = exact_match / len(val_labels)
    return acc, epoch_loss_valid_avg


def train(model, epoch, model_num, dtpath):
    train_features, train_graphs, train_labels = load_data(dtpath + trainset, dtpath + trainlabel,
                                                                        Dataset_Path + "pssm/", Dataset_Path + "graph/",Prot_Path+"prot/")
    val_features, val_graphs, val_labels = load_data(dtpath + testset, dtpath + testlabel,
                                                                Dataset_Path + "pssm/", Dataset_Path + "graph/",Prot_Path+"prot/")

    best_acc = 0
    best_epoch = 0
    cur_epoch = 0
    print("epoch:" + str(0))
    print("========== Evaluate Valid set ==========")
    valid_acc, epoch_loss_valid_avg = evaluate(model, val_features, val_graphs, val_labels)
    print("valid acc:", valid_acc)
    print("valid loss:", epoch_loss_valid_avg)
    # best_acc = valid_acc

    for epoch in range(epoch):
        model.train()
        for i in tqdm(range(len(train_labels))):

            sequence_features = torch.from_numpy(train_features[i])
            sequence_graphs = torch.from_numpy(train_graphs[i])

            labels = torch.from_numpy(np.array([int(float(train_labels[i]))]))

            sequence_features = torch.squeeze(sequence_features)
            sequence_graphs = torch.squeeze(sequence_graphs)


            if torch.cuda.is_available():
                features = sequence_features.cuda()
                graphs = sequence_graphs.cuda()

                y_true = labels.cuda().float()

            y_pred = model(features, graphs)
            loss = model.criterion(y_pred, y_true)
            loss /= BATCH_SIZE
            loss.backward()

            if (i % BATCH_SIZE == 0):
                model.optimizer.step()
                model.optimizer.zero_grad()

        print("epoch:" + str(epoch + 1))
        print("========== Evaluate Valid set ==========")
        valid_acc, epoch_loss_valid_avg = evaluate(model, val_features, val_graphs, val_labels )
        print("valid acc:", valid_acc)
        print("valid loss:", epoch_loss_valid_avg)
        if best_acc < valid_acc and epoch > 30:
            best_acc = valid_acc
            best_epoch = epoch + 1
            cur_epoch = 0
            torch.save(model.state_dict(), os.path.join('./model/best_model_' + model_num + '.pkl'))
        else:
            cur_epoch += 1
            if (cur_epoch > 50):
                break
    print("Best epoch at", str(best_epoch))
    print("Best acc at", str(best_acc))
    return best_epoch, best_acc


def main(model_name, data_path):
    model = Model()
    if to_generate_model_pkl_file:
        torch.save(model.state_dict(), os.path.join('./model/best_model_' + model_name + '.pkl'))
    model.load_state_dict(torch.load('./model/best_model_' + model_name + '.pkl'))
    if torch.cuda.is_available():
        model.cuda()
    epoch, acc = train(model, NUMBER_EPOCHS, model_name, data_path)
    f = open("train_result.txt", "a")
    f.write(str(epoch) + " acc : " + str(acc) + " \n")
    f.close()


if __name__ == "__main__":
    print("This is train.py")