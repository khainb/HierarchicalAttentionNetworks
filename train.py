import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_max_lengths,get_evaluation,get_pretrained_word_embedding,read_vocab
import argparse
import shutil
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from Dataset import Custom_Dataset
from HAN import HierarchicalAttention
import pandas as pd
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=13788)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=10,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/yelp_review_full_csv/train.csv")
    parser.add_argument("--test_set", type=str, default="data/yelp_review_full_csv/test.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="glove/glove.6B/glove.6B.200d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    device = torch.device("cuda" if (torch.cuda.is_available() ) else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    # max_word_length, max_sent_length = get_max_lengths(opt.train_set)
    max_word_length, max_sent_length=13,24
    vocab = read_vocab('data/yelp_review_full_csv/train.csv.txt')
    emb,word_to_ix= get_pretrained_word_embedding(opt.word2vec_path,vocab)
    df = pd.read_csv(opt.train_set,names=['label','text'])
    texts = np.array(df['text'])
    labels = np.array(df['label'])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(texts, labels):
        X_train, X_valid = texts[train_index], texts[test_index]
        y_train, y_valid = labels[train_index], labels[test_index]
    training_set = Custom_Dataset(X_train,y_train, word_to_ix, max_sent_length, max_word_length)
    valid_set = Custom_Dataset(X_valid,y_valid, word_to_ix, max_sent_length, max_word_length)
    training_generator = DataLoader(training_set,num_workers=32, **training_params)
    valid_generator = DataLoader(valid_set,num_workers=32, **training_params)
    df_test = pd.read_csv(opt.test_set,names=['label','text'])
    test_texts = np.array(df_test['text'])
    test_labels = np.array(df_test['label'])
    test_set = Custom_Dataset(test_texts,test_labels, word_to_ix, max_sent_length, max_word_length)
    test_generator = DataLoader(test_set,num_workers=32, **test_params)

    model =nn.DataParallel( HierarchicalAttention(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                       emb, max_sent_length, max_word_length))
    


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)


    if torch.cuda.is_available():
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True,min_lr=1e-8)
    best_acc=0.
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    for epoch in range(opt.num_epoches):
        print("Epoch "+str(epoch))
        for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.to(device)
                label = label.to(device)
            optimizer.zero_grad()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
        training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            opt.num_epoches,
            optimizer.param_groups[0]['lr'],
            loss, training_metrics["accuracy"]))

        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.to(device)
                    te_label = te_label.to(device)
                with torch.no_grad():
                    te_predictions = model(te_feature)
                te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            vl_loss_ls=[]
            vl_label_ls=[]
            vl_pred_ls=[]
            for vl_feature,vl_label in valid_generator:
                num_sample=len(vl_label)
                if torch.cuda.is_available():
                    vl_feature=vl_feature.to(device)
                    vl_label=vl_label.to(device)
                with torch.no_grad():
                    vl_predictions=model(vl_feature)
                vl_loss = criterion(vl_predictions, vl_label)
                vl_loss_ls.append(vl_loss * num_sample)
                vl_label_ls.extend(vl_label.clone().cpu())
                vl_pred_ls.append(vl_predictions.clone().cpu())
            vl_loss = sum(vl_loss_ls) / split
            vl_pred = torch.cat(vl_pred_ls, 0)
            vl_label = np.array(vl_label_ls)
            vl_metrics = get_evaluation(vl_label, vl_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])

            output_file.write(
                    "Epoch: {}/{} \nValid loss: {} Valid accuracy: {} \nValid confusion matrix: \n{}\nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    vl_loss,
                    vl_metrics["accuracy"],
                    vl_metrics["confusion_matrix"],
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {},Valid Loss: {}, Valid Accuracy: {}, Test Loss: {}, Test Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                vl_loss,vl_metrics["accuracy"],
                te_loss, test_metrics["accuracy"]))
            scheduler.step(vl_metrics["accuracy"])
            model.train()
            if vl_metrics["accuracy"] > best_acc:
                best_acc=vl_metrics["accuracy"]
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break


if __name__ == "__main__":
    opt = get_args()
    train(opt)
