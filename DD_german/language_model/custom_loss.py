from collections import OrderedDict
import pandas as pd
import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from sklearn.metrics import classification_report 
from collections import Counter
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, Trainer, AutoModelForSequenceClassification, TrainingArguments
from datasets import load_metric
labels =['<f/>','<e/>','<rps/>']
label2idx = {t:i for i, t in enumerate(labels)}
idx2label = {v: k for k, v in label2idx.items()}
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

def compute_metrics(eval_preds):
    "get accuracy and f1 score for each class"
    logits, label = eval_preds
    predictions = np.argmax(logits, axis=-1)
    report = classification_report(y_true=label, y_pred=predictions, target_names=['<f/>','<e/>','<rps/>'], output_dict=True)
    return {"report":str(report)}

class BertTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs,return_outputs=False):
        # compute loss here
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        sentence_rep = outputs[:1]
        logits = outputs.logits
        # THIS IS WHERE I HAVE PROBLME HOW TO COMPARE WHEATHER IT SHOUDLE BE DONE BETWEEN (SENTENCE AND ACTUAL CLASS) OR (ACTUAL CLASS AND PREDICTED CLASS)
        similarities = torch.nn.CosineSimilarity()
        print("------------->", similarities(sentence_rep, logits))
        # predicted_class = np.argmax(similarities)
        
        # loss = similarities[predicted_class]
        # outputs = np.argmax(similarities.detach().cpu())
        # print("-------------------->",loss, outputs)
        
        return (loss, outputs) if return_outputs else loss

def flatten(arrays:list) -> list:
    "convert 2-d array to 1-d array"
    return [x for xs in arrays for x in xs]

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

if __name__ == '__main__':
    model           = AutoModelForSequenceClassification.from_pretrained('dbmdz/bert-base-german-cased', num_labels=len(labels), classifier_dropout=0.1)
    tokenizer       = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')
    data_collator   = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = read_corpus('../data/word_tag_csv_word_dur2/')

    words = dataset['combine_col'].tolist() 
    labels = dataset['disflunecy_tag'].tolist()
        

    
    dataset_encoding = tokenizer(words, truncation=True, padding=True, return_tensors="pt")
    dataset = DisfluenyDataset(dataset_encoding, labels, label2idx)
    
    train_dataset, val_dataset, test_dataset = generate_train_test_valid_loader(dataset,  training_split=0.7, validation_split=0.15, train_batch_size=64)

    training_args  = TrainingArguments(
        output_dir='../saved_models/custom_loss',  # output directory
        logging_dir='../logs/custom_loss',  # directory for storing logs
        num_train_epochs=5,  # total number of training epochs
        per_device_train_batch_size=20,  # batch size per device during training
        per_device_eval_batch_size=8,  # batch size for evaluation
        learning_rate=2e-5,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_steps=100,
        save_total_limit=5,
        save_strategy='steps',
        save_steps=500,
        evaluation_strategy="epoch"
    ) 
    trainer = BertTrainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator)

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)
    print("metrics ================>", metrics)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

        # Evaluation

    print("*** Evaluate ***")
    metrics = trainer.evaluate()
    # metrics = fix_metrics(metrics)
    metrics["eval_samples"] = len(val_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)








    # print(vocab)
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
    # model = AutoModel.from_pretrained('bert-base-german-cased')
    # print("sentence, true label ------------------------->", words[1], labels[1])
    # sentence = words[1]
    # labels =['<f/>','<e/>','<rps/>']
    # inputs = tokenizer.batch_encode_plus([sentence] + labels,
    #                                  return_tensors='pt',
    #                                  pad_to_max_length=True)
    # input_ids = inputs['input_ids']
    # attention_mask = inputs['attention_mask']
    # # print(model(input_ids, attention_mask=attention_mask))
    # output = model(input_ids, attention_mask=attention_mask)[0]
    # sentence_rep = output[:1].mean(dim=1)
    # label_reps = output[1:].mean(dim=1)
    # similarities = F.cosine_similarity(sentence_rep, label_reps)
    # print("similarities", similarities, similarities[np.argmax(similarities.detach().cpu())])
    # closest = similarities.argsort(descending=True)
    # print("closed ", closest)
    # print("np.argmax", np.argmax(closest))
    # for ind in closest:
    #     print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')