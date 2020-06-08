import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from torchvision import datasets
import time
import os
from config import config
from models import MnistClassifier


def train_model(model, train, val, mini_batch_size):
    if not os.path.exists(config['models_dir']):
        os.mkdir(config['models_dir'])
    if not os.path.exists(config['logs_dir']):
        os.mkdir(config['logs_dir'])

    optimizer_name = config['optimizer']
    lr = config['lr']
    num_epochs = config['epochs']

    train_input, train_target = train[0].cuda(), train[1].cuda()
    val_input, val_target = val[0].cuda(), val[1].cuda()

    if optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss().cuda()
    num_samples = train_input.size(0)

    if mini_batch_size > num_samples:
        mini_batch_size = num_samples

    log_name = f"mnist_{optimizer_name}_batch{mini_batch_size}_lr{lr}.csv"
    log_path = os.path.join(config['logs_dir'], log_name)
    header = "epoch, sum_loss, avg_loss, val_loss, val_acc, train_acc, total_time"
    f = open(log_path, "x")
    f.write(header + "\n")
    # print(header)

    model_file_name = f"mnist_{optimizer_name}_batch{mini_batch_size}_lr{lr}.pt"
    model_path = os.path.join(config['models_dir'], model_file_name)

    tic = time.time()
    for e in range(1, num_epochs+1):
        sum_loss= 0
        count = 0
        for b in range(0, num_samples, mini_batch_size):
            b_end = min(b + mini_batch_size, num_samples)
            if b_end - b == 1:
                continue # avoid mini-batch of size 1 as we have batch-norm layers 
            train_input_mini_batch = train_input[b:min(b+mini_batch_size, num_samples)]
            train_target_mini_batch = train_target[b:min(b+mini_batch_size, num_samples)]
            optimizer.zero_grad()
            prediction_mini_batch = model(train_input_mini_batch)
            loss = criterion(prediction_mini_batch, train_target_mini_batch)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            count += 1

        toc = time.time()
        running_time = toc - tic
        avg_loss  =  sum_loss / count

        val_prediction = model(val_input)
        predicted_labels = torch.argmax(val_prediction, dim=1)
        val_loss = criterion(val_prediction, val_target).item()
        val_acc = (predicted_labels == val_target).float().mean().item()
        
        train_acc = test_model(model, train_input, train_target)

        f.write(f"{e}, {sum_loss}, {avg_loss}, {val_loss}, {val_acc}, {train_acc}, {running_time}\n")
        if (e == 1 or e == num_epochs or e % config['verbosity_mod'] == 0):
            # print(f"{e}, {sum_loss}, {avg_loss}, {val_loss}, {val_acc}, {train_acc}, {running_time}")
            print(f"epoch {e}, batch_size {mini_batch_size}, train_acc {train_acc}, time {running_time}")

    f.close()

    save_state(path=model_path, model=model, optimizer=optimizer, epoch=num_epochs, 
            mini_batch_size=mini_batch_size, loss=loss, lr=lr)

    return model

def test_model(model, test_input, test_target):
    prediction = model(test_input)
    predicted_labels = torch.argmax(prediction, dim=1)
    accuracy = (predicted_labels == test_target).float().mean().item()
    return accuracy

def save_state(path, model, optimizer, epoch, mini_batch_size, loss, lr):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate': lr,
        'mini_batch_size': mini_batch_size
    }
    torch.save(state, path)

def run_mini_batch_experiment(train, val):
    mini_batch_size = 2
    i = 1
    while mini_batch_size < train[0].size(0)
        if i % config['num_gpus'] == config['gpu_id'] or not config['split_task']:
            print(f"Training with mini-batch size {mini_batch_size}")
            train_model(MnistClassifier().cuda(), train, val, mini_batch_size=mini_batch_size)
            print()
        i += 1
        mini_batch_size = 2 ** i
