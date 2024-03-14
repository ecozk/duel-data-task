import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, Trainer, AutoModelForSequenceClassification, TrainingArguments
from datasets import load_metric
from sklearn.metrics import classification_report 
labels =['<f/>','<e/>','<rps/>']
label2idx = {t:i for i, t in enumerate(labels)}
idx2label = {v: k for k, v in label2idx.items()}
import wandb
import warnings
warnings.filterwarnings("ignore")
wandb.init(
    # set the wandb project where this run will be logged
    project="german-disflueney",
)
def compute_metrics(eval_preds):
    "get accuracy and f1 score for each class"
    # metric1 = load_metric("precision")
    # metric2 = load_metric("recall")
    # metric3 = load_metric("f1")
    # metric4 = load_metric("accuracy")
    
    logits, label = eval_preds
    predictions = np.argmax(logits, axis=-1)
    # precision = metric1.compute(predictions=predictions, references=label, average="micro")["precision"]
    
    # recall = metric2.compute(predictions=predictions, references=label, average="micro")["recall"]
    
    # f1 = metric3.compute(predictions=predictions, references=label, average="micro")["f1"]
    
    # accuracy = metric4.compute(predictions=predictions, references=label)["accuracy"]

    report = str(classification_report(y_true=label, y_pred=predictions, target_names=['<f/>','<e/>','<rps/>'], output_dict=True))
    return {"report":report}
def create_embedding(dataframe:pd.DataFrame) -> None:
    CONTEXT_SIZE = 2
    EMBEDDING_DIM = 10
    "create embedding for our dataset"
    words = dataframe['word'].tolist()
    vocab = set(words)
    # print(vocab,len(vocab), len(words))
    ngrams = [
        (
            [words[i-j-1] for j in range(CONTEXT_SIZE)], words[i]
        )
        for i in range(CONTEXT_SIZE, len(words))
    ]
    print(ngrams)
    word2id = {w:i for i, w in enumerate(words)}
    id2word = {i:w for i, w in enumerate(words)}
    n_class = len(word2id)
    print(n_class)
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
        df_temp = pd.read_csv(os.path.join(dir_path,file), names=col_names)
        df_append = pd.concat([df_append, df_temp])
    df_append['combine_col'] = df_append['word'] + '[SEP]'+ df_append['pos_tag']
    return df_append

def preprocessing(table:pd.DataFrame)-> pd.DataFrame:
    # e class has the smallest number of sample size of
    sample_size = min(table['disflunecy_tag'].value_counts())
    
    f_tag_table = table.loc[table['disflunecy_tag'] == '<f/>'][0:sample_size]
    e_tag_table = table.loc[table['disflunecy_tag'] == '<e/>'][0:sample_size]
    rmps_tag_table = table.loc[table['disflunecy_tag'] == '<rps/>'][0:sample_size]
    
    held_out_f_table = table.loc[table['disflunecy_tag'] == '<f/>'][sample_size:]
    held_out_e_table = table.loc[table['disflunecy_tag'] == '<e/>'][0:sample_size]
    held_out_rmp_table = table.loc[table['disflunecy_tag'] == '<rps/>'][sample_size:]
    return pd.concat([f_tag_table, e_tag_table, rmps_tag_table]), pd.concat([held_out_f_table, held_out_e_table, held_out_rmp_table])


def generate_train_test_valid_loader(disfluency_dataset, training_split, validation_split, train_batch_size):
    print(f"Generation train and validaiton loader for training for number of samples {len(disfluency_dataset)}")
    total_size    = len(disfluency_dataset)
    train_size    = int(training_split * total_size)

    # val_size      = int (validation_split*total_size)
    val_size      = total_size - train_size
    # test_size = total_size - train_size - val_size

    # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(disfluency_dataset, [train_size, val_size, test_size])
    train_dataset, val_dataset = torch.utils.data.random_split(disfluency_dataset, [train_size, val_size])


    print(f"length of training dataset {len(train_dataset)}")
    print(f"length of validation dataset {len(val_dataset)} ")
    
    # print(f"length of test dataset {len(test_dataset)}")
    # train_loader        = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size)
    # validation_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=train_batch_size)
    # test_loader         = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    return train_dataset, val_dataset
def test(held_out_dataset):
    print("------------------------------------------------------- Testing loop on held out data-------------------------------------------")
    model = AutoModelForSequenceClassification.from_pretrained('../saved_models/bert_with_pos_tag_final_trail')
    # tokenizer = AutoTokenizer.from_pretrained('../saved_models/bert_with_pos_tag/')
    y_true = list()
    y_pred = list()
    for index, batch in enumerate(held_out_dataset):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        y_true.append(batch['labels'].item())
        with torch.no_grad():
            logits = model(**batch).logits
            print(model(**batch))
        predictions = np.argmax(logits.detach().numpy())
        y_pred.append(predictions)
        # print(f"procesing index {index} out of {len(held_out_dataset)}, {batch['labels'].item()}, {predictions}")
    print(classification_report(y_true=y_true, y_pred=y_pred,  target_names=['<f/>','<e/>','<rps/>'],zero_division=0))
    return

if __name__ == '__main__':
    model           = AutoModelForSequenceClassification.from_pretrained('dbmdz/bert-base-german-cased', num_labels=len(labels), classifier_dropout=0.1)
    tokenizer       = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')
    data_collator   = DataCollatorWithPadding(tokenizer=tokenizer)



    df = read_corpus('../data/word_tag_csv_word_dur2/')
    train_dataframe, held_out_dataset = preprocessing(df)

    train_words = train_dataframe['combine_col'].tolist() 
    train_labels = train_dataframe['disflunecy_tag'].tolist()

    train_dataset_encoding  = tokenizer(train_words, truncation=True, padding=True, return_tensors="pt")
    train_dataset           = DisfluenyDataset(train_dataset_encoding, train_labels, label2idx)

    # train_loader        = torch.utils.data.DataLoader(train_dataset, batch_size=64)
    train_dataset, val_dataset = generate_train_test_valid_loader(train_dataset,  training_split=0.7, validation_split=0.15, train_batch_size=64)
    

    held_out_words = held_out_dataset['combine_col'].tolist()
    held_out_label = held_out_dataset['disflunecy_tag'].tolist()

    held_out_encoding = tokenizer(held_out_words, truncation=True, padding=True, return_tensors="pt")

    held_out_dataset = DisfluenyDataset(held_out_encoding, held_out_label, label2idx)

    held_out_loader  = torch.utils.data.DataLoader(held_out_dataset, batch_size=1, shuffle=True)
  
    # training_args  = TrainingArguments(
    #     output_dir='../saved_models/bert_with_pos_tag_final_trail',  # output directory
    #     logging_dir='../saved_models/bert_with_pos_tag_final_trail',  # directory for storing logs
    #     num_train_epochs=5,  # total number of training epochs
    #     per_device_train_batch_size=20,  # batch size per device during training
    #     per_device_eval_batch_size=8,  # batch size for evaluation
    #     learning_rate=2e-5,
    #     warmup_steps=500,  # number of warmup steps for learning rate scheduler
    #     weight_decay=0.01,  # strength of weight decay
    #     logging_steps=100,
    #     save_total_limit=5,
    #     save_strategy='epoch',
    #     save_steps=500,
    #     evaluation_strategy="steps",
    #     report_to="wandb"
    # ) # Evaluate at very logging steps
    
    # trainer = Trainer(
    #     model=model,  # the instantiated Transformers model to be trained
    #     args=training_args,  # training arguments, defined above
    #     train_dataset=train_dataset,  # training dataset
    #     eval_dataset=val_dataset,  # evaluation dataset
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    #     data_collator=data_collator)

    # train_result = trainer.train()
    # trainer.save_model()  # Saves the tokenizer too for easy upload
    # metrics = train_result.metrics

    # metrics["train_samples"] = len(train_dataset)
    # print("metrics ================>", metrics)
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()

    #     # Evaluation

    # print("*** Evaluate ***")
    # metrics = trainer.evaluate()
    # # metrics = fix_metrics(metrics)
    
    # metrics["eval_samples"] = len(val_dataset)
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    test(held_out_loader)
    # print(trainer.state.log_history)
