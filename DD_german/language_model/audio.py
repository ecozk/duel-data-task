import os
import time
import pandas as pd
import librosa
import numpy as np
import h5py
from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from multiprocessing import Process
from imblearn.over_sampling import SMOTE
from torch.utils.tensorboard import SummaryWriter

labels =['<f/>','<e/>','<rps/>']
label2idx = {t:i for i, t in enumerate(labels)}
idx2label = {v: k for k, v in label2idx.items()}
def get_class_count(dataset):
    # num_classes = len(labels)
    # label = torch.zeros(num_classes, dtype=torch.long)
    # for _, target in dataset:
    #     label += target
    #     print(label)
    # return label
    class_count = {'<f/>' : 0, '<e/>': 0, '<rps/>' : 0}
    # for _, label in dataset:
    #     idx = torch.argmax(label).item()
    #     # print(label, idx)
    #     label = idx2label[idx]
    #     class_count[label] = class_count[label]+1
    #     # print(class_count)
    # return class_count
    return None
def generate_train_test_valid_loader(disfluency_dataset, training_split, validation_split, train_batch_size):
    print(f"Generation train and validaiton loader for training for number of samples {len(disfluency_dataset)}")
    total_size    = len(disfluency_dataset)
    train_size    = int(training_split * total_size)
    val_size      = int (validation_split*total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(disfluency_dataset, [train_size, val_size, test_size])
    # train_dataset, test_dataset = torch.utils.data.random_split(disfluency_dataset, [train_size, test_size])
    
    print(f"length of training dataset {len(train_dataset)} and class count is {get_class_count(train_dataset)}")
   
    print(f"length of validation dataset {len(val_dataset)} and class count is {get_class_count(val_dataset)}")
    print(f"length of test dataset {len(test_dataset)} and class count is {get_class_count(test_dataset)}")

    train_loader        = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
    validation_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=train_batch_size)
    test_loader         = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    return train_loader,test_loader , validation_loader

def read_corpus(dir_path:any) -> pd.DataFrame:
    "function for reading all csv file and generating dataset of it"
    df_append = pd.DataFrame()
    col_names =['Participant', 'utterance number', 'utterance length', 'word number' ,'start_time', 'end_time', 'word durnation', 'word', 'pos_tag','disflunecy_tag']
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    for file in csv_files:
        # print(f"processing file {file}")
        df_temp = pd.read_csv(os.path.join(dir_path,file),names=col_names)
        df_append = pd.concat([df_append, df_temp])
    # print(df_append.head())
    return df_append


def get_stats(data:pd.DataFrame) -> int:
    "state for word durnation"
    # print("---------------------stats of word durnation and other column are as follow---------------------------------")
    # print(data.describe())
    larget_time_durnation = data['word durnation'].max()
    min_time_durnation = data['word durnation'].min()
    return larget_time_durnation, min_time_durnation

def generate_melSpectrogram(df:pd.DataFrame, session:int, n_mels=512)-> None:
    session_data= df.loc [ (df['Participant'] == str(session)+'a') | (df['Participant'] == str(session)+'b') ] # number of sample for each session for both speaker
    print(f"total number of sample for session {session} are {len(session_data)}")
    audio_file_path = f"../data/audio/r{session}/r{session}.wav"
    print(audio_file_path)
    audio_data, sample_rate = librosa.load(audio_file_path)
    mel_spectrogram = list()
    target = list()
    for index, rows in session_data.iterrows():
        start_time = rows['start_time']
        end_time = rows['end_time']
        if rows['disflunecy_tag'] == '<f/>':
            target.append([1,0,0])
        elif rows['disflunecy_tag'] == '<e/>':
            target.append([0,1,0])
        elif rows['disflunecy_tag'] == '<rps/>':
            target.append([0,0,1])

        audio_data_interval = audio_data[int(start_time * sample_rate):int(end_time * sample_rate)]

        spectrogram = librosa.feature.melspectrogram(y=audio_data_interval, sr=sample_rate, n_mels=n_mels, n_fft=512)

        y_left = n_mels - spectrogram.shape[1]

        padded_mel = np.pad(spectrogram, ((0,0), (0,y_left) ), mode='constant', constant_values=0)

        mel_spectrogram.append(padded_mel.reshape(1,512,512))
    return np.array(mel_spectrogram), np.array(target)

def preprocessing(dataset:pd.DataFrame) -> None:
    if os.path.exists("dataset.h5"):
        os.remove("dataset.h5")
    f = h5py.File('dataset.h5', 'a')
    for i in range(1, 10):
        new_data, new_label = generate_melSpectrogram(dataset, i)
        print(f"number of spectrogam generated are {(new_data.shape)}, {(new_label.shape)}")
        if i == 1:
            f.create_dataset('data', data=new_data, compression="gzip", chunks=True, maxshape=(None,1, 512,512))
            f.create_dataset('label', data=new_label, compression="gzip", chunks=True, maxshape=(None,3)) 
         # Append new data to it
        else:
            f['data'].resize((f['data'].shape[0] + new_data.shape[0]), axis=0)
            f['data'][-new_data.shape[0]:] = new_data
            f['label'].resize((f['label'].shape[0] + new_label.shape[0]), axis=0)
            f['label'][-new_label.shape[0]:] = new_label
        print("I am on iteration {} and 'data' chunk has shape:{}".format(i,f['data'].shape))
    f.close()

    # print("--------------------------- Over sampling using SMOTE---------------------------")
    # sm = SMOTE(random_state=42)
    # h5_file = h5py.File('dataset.h5')
    # X = h5_file['data'][:]
    # y = h5_file['label'][:]
    # Y = np.zeros(y.shape[0])
    # for index, value in enumerate(y):
    #     Y[index] = np.argmax(value)
    # n_sample, n_channal, height, width = X.shape
    # X = X.reshape(n_channal*n_sample, height*width)
    # print(f"Before oversampling {y.shape}{X.shape}")
    # X_res, y_res = sm.fit_resample(X, Y)
    # print(f'After oversampling {Y.shape} {X.shape}')


class AudioDataset(Dataset):
    def __init__(self, h5_paths):
        super(AudioDataset, self).__init__()
        self.h5_file = h5py.File(h5_paths, 'r')
        self.melspectrogram = self.h5_file['data']
        self.label = self.h5_file['label']
    def __getitem__(self, index):
        return ([
            torch.from_numpy(self.melspectrogram[index]),
            torch.from_numpy(self.label[index])
        ])
    def __len__(self):
        return self.melspectrogram.shape[0]
    def __del__(self):
        self.h5_file.close()

class BaselineCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2,2)
        )
        self.classifier = nn.Sequential(
            torch.nn.Linear(246016,1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.audio_feature_extractor(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)
        return x
    
def train(train_loader, val_loader, model, loss_function, optimizer, epochs, device, writer):
    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))
    
    history = {}
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []
    start_time_sec = time.time()
    for epoch in range(epochs):
        model=model.to(device)
        model.train()
        train_loss = 0.0
        num_train_correct = 0
        num_train_examples = 0
        total_step = len(train_loader)
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            images = batch[0].to(device)
            target = batch[1].to(device)

            output = model(images.float())
            # print(output, target)
            loss = loss_function(output, target.float())

            loss.backward()
            optimizer.step()

            num_train_correct  += (torch.argmax(output, 1) == torch.argmax(target,1)).float().sum()
            num_train_examples += images.shape[0]

            train_loss +=loss.item()

            if (i+1)%10==0:
                avg_loss = train_loss/10
                accuracy = (num_train_correct/num_train_examples)*100
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Training Loss', avg_loss, global_step)
                writer.add_scalar('Training Accuracy', accuracy, global_step)

        epoch_acc =  (num_train_correct / num_train_examples).item()
        epoch_loss  = train_loss / len(train_loader.dataset)
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        writer.add_scalar('Epoch Training Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch Training Accuracy', epoch_acc)
        print(f'Epoch {epoch+1}/{epochs}, train loss: {epoch_loss}, train acc: {epoch_acc}' )
        # --- EVALUATE ON VALIDATION SET -------------------------------------

        # model.eval()
        # val_loss       = 0.0
        # num_val_correct  = 0
        # num_val_examples = 0

        # for batch in val_loader:
        #     images = batch[0].to(device)
        #     target = batch[1].to(device)

        #     output = model(images.float())
        #     loss = loss_function(output, target.float())

        #     val_loss         += float(loss.data.item()*images.size(0))
        #     # print(f"validatio loss {val_loss}")
        #     num_val_correct  += (torch.argmax(output, 1) == torch.argmax(target,1)).sum().item()
        #     num_val_examples += images.shape[0]
        # val_acc  = num_val_correct / num_val_examples
        # val_loss = val_loss / len(val_loader.dataset)

        # print('Epoch %3d/%3d, train loss: %5.2f, train acc: %5.2f, val loss: %5.2f, val acc: %5.2f' % \
        #       (epoch+1, epochs, train_loss, train_acc, val_loss, val_acc))
        # # print(f'Epoch {epoch+1}/{epochs}, train loss: {train_loss}, train acc: {train_acc}' )

      
        # history['val_loss'].append(val_loss)
        
        # history['val_acc'].append(val_acc)

         # END OF TRAINING LOOP
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss_function,
        #     }, '../saved_models/audio.pt')
    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

    return history
def test(model, test_loader, device):
    # checkpoint = torch.load('../saved_models/audio.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()
    y_true = []
    y_pred = []
    for samples in test_loader:
        y_true.append(torch.argmax(samples[1]).item())
        pred = model(samples[0].to(device).float())
        y_pred.append(torch.argmax(pred).item())
    print("---------------------------------------------------------------------------------------------------------------")
    print(classification_report(y_true, y_pred, target_names=labels))
if __name__ == '__main__':
    # data_path = '../data/word_tag_csv_word_dur2/'
    # dataset = read_corpus(data_path)
    # larget, minimunm = get_stats(dataset)
    # preprocessing(dataset=dataset)
    audio_dataset = AudioDataset('dataset.h5')
    batch_size = 64 
    validation_split = .2
    shuffle_dataset = True
    train_loader, test_loader, val_loader = generate_train_test_valid_loader(
        disfluency_dataset=audio_dataset, 
        training_split=0.7, 
        validation_split=0.15, 
        train_batch_size=batch_size
    )
    torch.cuda.empty_cache()
    writer = SummaryWriter("../saved_model/audio_exp1")
    # device = "cuda" if torch.cuda.is_available () else "cpu"
    device = "cpu"
    model = BaselineCNN()
    calculate_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 1
    history = train(train_loader, val_loader, model, calculate_loss, optimizer, epochs, device, writer)
    print(history)
    writer.close()
    test(model, test_loader, device)