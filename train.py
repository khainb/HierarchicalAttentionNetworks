import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_max_lengths,get_evaluation,get_pretrained_word_embedding,read_vocab
import argparse
import shutil
import numpy as np

from Dataset import Custom_Dataset
from HAN import HierarchicalAttention
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=100)
    parser.add_argument("--sent_hidden_size", type=int, default=100)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/yelp_review_full_csv/train.csv")
    parser.add_argument("--test_set", type=str, default="data/yelp_review_full_csv/test.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="glove/glove.6B/glove.6B.100d.txt")
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
    training_set = Custom_Dataset(opt.train_set, word_to_ix, max_sent_length, max_word_length)
    training_generator = DataLoader(training_set, **training_params)
    test_set = Custom_Dataset(opt.test_set, word_to_ix, max_sent_length, max_word_length)
    test_generator = DataLoader(test_set, **test_params)

    model =nn.DataParallel( HierarchicalAttention(opt.word_hidden_size, opt.sent_hidden_size, opt.batch_size, training_set.num_classes,
                       emb, max_sent_length, max_word_length))
    


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)


    if torch.cuda.is_available():
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    best_loss = 1e5
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
            num_iter_per_epoch,
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
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break


if __name__ == "__main__":
    opt = get_args()
    train(opt)
