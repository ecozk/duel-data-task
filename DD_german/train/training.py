import torch
import numpy as np
import torch
import os
import time
import logging
import json
from model import DeepDisfluencyTagger, DisfluencyDataset
from utils import plot_figure, EarlyStopping
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
hyperparameter = {
    "experiment_dir"    :"./withMotionData/lstmWithMLP",
    "batch_size"        :64,
    "training_split"    :0.7,
    "validation_split"  :0.15,
    "epochs"            :100    ,
    "learning_rate"     :0.001,
    "num_classes"       :3,
    "lstm_layers"       :2,
    "device"            : "cuda" if torch.cuda.is_available() else "cpu",
    "input_size"        : (1004),
    "hidden_size"       : 100,
    "dropout"           : 0.3,
    "seq_length"        : 1004
}

def generate_train_test_valid_loader(disfluency_dataset):
    print(f"Generation train and validaiton loader for training for number of samples {len(disfluency_dataset)}")
    total_size    = len(disfluency_dataset)
    train_size    = int(hyperparameter['training_split'] * total_size)
    val_size      = int (hyperparameter['validation_split']*total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(disfluency_dataset, [train_size, val_size, test_size])
    
    train_loader        = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameter["batch_size"])
    validation_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=hyperparameter["batch_size"])
    test_loader         = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparameter['batch_size'])

    return train_loader, validation_loader, test_loader



def save_network(epoch_label, model, optimizer, calculate_loss):
    "method to save model and optimizer state at checkpoun"
    save_filename = 'net_%s.pth' % epoch_label
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(hyperparameter['experiment_dir'], save_filename)
    print(f"saving network state in {save_path}")
    torch.save({'epoch': epoch_label,'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),'loss': calculate_loss,}, 
                save_path
            )
    return

def fit(train_loader, model, loss_function, optimizer, device):
    model = model.to(device)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for index, batch in enumerate(train_loader):
        optimizer.zero_grad()
        feature_vector = batch[0].to(device)
        target_vector = batch[1].to(device)

        output = model(feature_vector.float())
        loss = loss_function(output, target_vector.float())
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*feature_vector.size(0)
        
        correct += (torch.argmax(output, 1) == torch.argmax(target_vector, 1)).float().sum()
        total += feature_vector.size(0)
    
    train_loss /= len(train_loader)
    train_acc = correct.cpu()/total
    return train_loss, train_acc

def evaluate(val_loader, model, loss_function, optimizer, device):
    model=model.to(device)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(val_loader):
        feature_vector = batch[0].to(device)
        target_vector = batch[1].to(device)
        
        output = model(feature_vector.float())
        loss = loss_function(output, target_vector.float())
        
        val_loss += loss.item()*feature_vector.size(0)
        correct += (torch.argmax(output, 1) == torch.argmax(target_vector, 1)).float().sum()

        total += feature_vector.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct.cpu() / total
    return val_loss, val_accuracy


def train(train_loader,val_loader, model, loss_function, optimizer, epochs, device):
    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % (type(model).__name__, type(optimizer).__name__, optimizer.param_groups[0]['lr'], epochs, device))
    
    history = {}
    history['train_loss'] = []
    history['val_loss'] = []
    history['train_acc'] = []
    history['val_acc'] = []
    early_stopping = EarlyStopping(patience=3, verbose=True)
    start_time_sec = time.time()
    for epoch in range(epochs):
        
        train_loss, train_acc = fit(train_loader, model, loss_function, optimizer, device)
        val_loss, val_acc = evaluate(val_loader, model, loss_function, optimizer, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        early_stopping(history['val_loss'][-1])
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        save_network(epoch_label=epoch, model=model, optimizer=optimizer, calculate_loss=loss_function)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f},Val Loss:{val_loss:.4f}, Val Acc:{val_acc:.4f}")
        print()

    
    
    
    total_time_sec = time.time() - start_time_sec
    print(f"Time total: {total_time_sec}")
    return history

if __name__ == '__main__':
    with open('config.json', 'w') as f:
        json.dump(hyperparameter, f)    
    model = DeepDisfluencyTagger(hyperparameter["num_classes"], hyperparameter['input_size'], 
                                 hyperparameter["hidden_size"],hyperparameter["lstm_layers"], 
                                 hyperparameter['seq_length'],hyperparameter['dropout'], hyperparameter['device']
                                )
    
    if os.path.isdir(hyperparameter['experiment_dir']) is False:
        # print("Setting up dir...")
        os.makedirs(hyperparameter['experiment_dir'])
    logging.basicConfig(filename = hyperparameter['experiment_dir']+'/logs.log', level = logging.DEBUG, format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    
    disflueny_data = DisfluencyDataset('./disf_tags_word_level_w_motion/dataset.h5py')

    train_loader, validation_loader, test_loader = generate_train_test_valid_loader(disfluency_dataset=disflueny_data)
    
    optimizer = optimizer = torch.optim.Adam(model.parameters(), hyperparameter['learning_rate'])
    
    calculate_loss = torch.nn.CrossEntropyLoss()

    history = train(train_loader, validation_loader, model, calculate_loss, optimizer, hyperparameter['epochs'], hyperparameter['device'])
    plot_figure(history)
    
    # sessions = range(1, 10)
    # participant = ["a", "b"]
    # for session in sessions:
    #     for par in participant:
    #         word_level_file = "./tag_csv_partials_to_be_removed/"+str(session)+par+".h5py"
    #         logging.debug(f"feeding data from file {word_level_file}")
    #         print(f"feeding data from file {word_level_file}")
    #         outputfile = str(session)+par
    #         disflueny_data = DisfluencyDataset(word_level_file)
    #         train_loader, validation_loader = generate_train_valid_loader(disfluency_dataset=disflueny_data)
    #         history = train(train_loader=train_loader, val_loader=validation_loader, model=model, loss_function=calculate_loss, optimizer=optimizer, epochs=hyperparameter['epochs'], device=hyperparameter['device'])
    #         plot_figure(history, outputfile)