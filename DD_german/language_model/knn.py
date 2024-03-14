import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, Trainer, AutoModelForSequenceClassification, TrainingArguments
from datasets import load_metric
from sklearn.metrics import classification_report 
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
# from torchtext.data.utils import get_tokenizer

labels =['<f/>','<e/>','<rps/>']
label2idx = {t:i for i, t in enumerate(labels)}
idx2label = {v: k for k, v in label2idx.items()}
import wandb
def train():
    pass
def validation():
    pass
def test():
    pass
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="german-disflueney",
# )
def compute_metrics(eval_preds):
    "get accuracy and f1 score for each class"
    logits, label = eval_preds
    predictions = np.argmax(logits, axis=-1)
    report = str(classification_report(y_true=label, y_pred=predictions, target_names=['<f/>','<e/>','<rps/>'], output_dict=True))
    return {"report":report}
def flatten(arrays:list) -> list:
    "convert 2-d array to 1-d array"
    return [x for xs in arrays for x in xs]
def create_embedding(dataframe:pd.DataFrame) -> None:
    "create embedding for our dataset"
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 600
    embedding_model = WordEmbeddings('de')
    words = dataframe['word'].tolist()
    pos_tags = dataframe['pos_tag'].tolist()
    vocab = set(words+pos_tags)
    ngrams = list()
    for i in range(CONTEXT_SIZE, len(words)):
        temp_list = list()
        for j in range(CONTEXT_SIZE):
            temp_list.append(words[i-j-1]+' '+pos_tags[i-j-1])
        temp_list.append(words[i]+' '+pos_tags[i])
        ngrams.append(" ".join(temp_list))
    # ngrams = [
    #     (
    #         [words[i-j-1] for j in range(CONTEXT_SIZE)], words[i]
    #     )
    #     for i in range(CONTEXT_SIZE, len(words))
    # ]
    embedding = dict()
    for index, word in enumerate(vocab):
        temp_embedding = list()
        sen = Sentence(word)
        embedding_model.embed(sen)
        for token in sen:
            temp_embedding.append(token.embedding.tolist())
        temp_embedding = flatten(temp_embedding)
        if len(temp_embedding) <= EMBEDDING_DIM:
            temp_embedding = temp_embedding+[0]*(EMBEDDING_DIM-len(temp_embedding))
        embedding[word] = temp_embedding
    # print (embedding)
    word2id = {w:i for i, w in enumerate(vocab)}
    id2word = {i:w for i, w in enumerate(vocab)}
    # print("----------------->", n_class, len(id2word))
    return len(vocab), word2id, id2word, embedding



class DisfluenyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, tag2idx):
        self.encodings = encodings
        self.labels = labels
        self.tag2idx = tag2idx
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.tag2idx[self.labels[idx]])
        return item 
    def __len__(self):
        return len(self.labels)
    
def read_corpus(dir_path:any) -> pd.DataFrame:
    "function for reading all csv file and generating dataset of it"
    df_append = pd.DataFrame()
    col_names =['Participant', 'utterance number', 'utterance length', 'word number' ,'start_time', 'end_time', 'word durnation', 'word', 'pos_tag','disflunecy_tag']
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    for file in csv_files:
        # print(f"processing file {file}")
        df_temp = pd.read_csv(os.path.join(dir_path,file),names=col_names)
        df_append = pd.concat([df_append, df_temp])
    df_append['combine_col'] = df_append['word'] + '[SEP]'+ df_append['pos_tag'] 
    # print(df_append.head())
    return df_append

def preprocessing(table:pd.DataFrame)-> pd.DataFrame:
    sample_size = min(table['disflunecy_tag'].value_counts())
    # print(table['disflunecy_tag'].value_counts())
    f_tag_table = table.loc[table['disflunecy_tag'] == '<f/>'].head(sample_size)
    e_tag_table = table.loc[table['disflunecy_tag'] == '<e/>'].head(sample_size)
    rmps_tag_table = table.loc[table['disflunecy_tag'] == '<rps/>'].head(sample_size)
    # print(f_tag_table)
    return pd.concat([f_tag_table, e_tag_table, rmps_tag_table])
def generate_train_test_valid_loader(disfluency_dataset, training_split, validation_split, train_batch_size):
    print(f"Generation train and validaiton loader for training for number of samples {len(disfluency_dataset)}")
    total_size    = len(disfluency_dataset)
    train_size    = int(training_split * total_size)
    val_size      = int (validation_split*total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(disfluency_dataset, [train_size, val_size, test_size])
    # train_dataset, test_dataset = torch.utils.data.random_split(disfluency_dataset, [train_size, test_size])
    print(f"length of training dataset {len(train_dataset)}")
   
    print(f"length of validation dataset {len(val_dataset)} ")
    print(f"length of test dataset {len(test_dataset)}")

    # train_loader        = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
    # validation_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=train_batch_size)
    # test_loader         = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    return train_dataset, val_dataset, test_dataset
class NNLM(torch.nn.Module):
    def __init__(self, embedding_matrix, embedding_dim, word2id, id2word):
        super(NNLM,self).__init__()
        self.word2idx = word2id
        self.idx2word = id2word
        self.embedding = torch.nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix)) # embedding are updated every time 
        self.embedding.weight.requires_grad = True
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=100, num_layers=1, batch_first=True) # for nlp feature extraction
        self.cnn_block = torch.nn.Sequential() # for audio feature extraction
        # self.fc = torch.nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        index = torch.tensor(self.word2idx[x], dtype=int)
        x = self.embedding(index)
        print("------------------> output of embedding layer", x.shape)
        x = self.lstm(x.reshape((1, 600)))
        print("output of lstm", x)
        return x

if __name__ == '__main__':
    df = read_corpus('../data/word_tag_csv_word_dur2')
    dataset = preprocessing(df)
    vocab_size, word2id, id2word, embedding = create_embedding(dataset)

    vocab_list = list(embedding.keys())
    
    embedding_dim = len(embedding[vocab_list[0]])
    
    embedding_matrix = torch.zeros(len(vocab_list), embedding_dim)
    
    # Iterate through the vocabulary list and populate the embedding matrix with the pre-trained embeddings
    for i, word in enumerate(vocab_list):
        if word in embedding:
            embedding_matrix[i] = torch.tensor(embedding[word])
    
    model = NNLM(embedding_matrix, embedding_dim, word2id, id2word)
    
    sample = dataset['word'].tolist()[2]
    
    model(sample)
   