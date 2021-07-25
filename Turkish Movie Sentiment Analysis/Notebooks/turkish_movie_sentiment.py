# Importing stock libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing Hugging Package
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast, RobertaForSequenceClassification

# Importing wand
import wandb

# Setting up the accelerators

# # GPU
# from torch import cuda
# device = 'cuda' if cuda.is_available() else 'cpu'

# TPU
import torch_xla
import torch_xla.core.xla_model as xm
device = xm.xla_device()

data = pd.read_csv("turkish-movie-sentiment-analysis-dataset/turkish_movie_sentiment_dataset.csv")
data['point'] = data['point'].str.replace(',', '.').astype(float)

def to_sentiment(rating):
  if rating <= 2.5:
    return 0
  elif 4 > rating > 2.5:
    return 1
  else:
    return 2

data['sentiment'] = data.point.apply(to_sentiment)

class_names = ['negative', 'neutral', 'positive']

ax = sns.countplot(x=data.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names);

import re

TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # converting turkish characters to international ex. ı to i
    sentence = re.sub('ı', "i", sentence)
    sentence = re.sub('ç', "c", sentence)
    sentence = re.sub('ö', "o", sentence)
    sentence = re.sub('ğ', "g", sentence)
    sentence = re.sub('ü', "u", sentence)
    sentence = re.sub('ş', "s", sentence)

    # Removing newlines
    sentence = re.sub(r'\n', ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


data["comment"] = data["comment"].apply(lambda x: preprocess_text(str(x)))
data["comment"] = data["comment"].apply(lambda x: ' '.join([w.strip() for w in x.split() if w != ' ']))


data = data[['comment','sentiment']]


class Preprocess:
    def __init__(self, df):
        """
        Constructor for the class
        :param df: Input Dataframe to be pre-processed
        """
        self.df = df
        self.encoded_dict = dict()

    def encoding(self, x):
        if x not in self.encoded_dict.keys():
            self.encoded_dict[x] = len(self.encoded_dict)
        return self.encoded_dict[x]

    def processing(self):
        self.df['encoded_polarity'] = self.df['sentiment'].apply(lambda x: self.encoding(x))
        self.df.drop(['sentiment'], axis=1, inplace=True)
        return self.encoded_dict, self.df



class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data.comment[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.encoded_polarity[index], dtype=torch.float)
        }

    def __len__(self):
        return self.len


# Creating a function that returns the dataloader based on the dataframe and the specified train and validation batch size.

def return_dataloader(df, tokenizer, train_batch_size, validation_batch_size, MAX_LEN, train_size=0.7):
    train_size = 0.7
    train_dataset = df.sample(frac=train_size, random_state=200)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(val_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    validation_set = CustomDataset(val_dataset, tokenizer, MAX_LEN)

    train_params = {'batch_size': train_batch_size,
                    'shuffle': True,
                    'num_workers': 1
                    }

    val_params = {'batch_size': validation_batch_size,
                  'shuffle': True,
                  'num_workers': 1
                  }

    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **val_params)

    return training_loader, validation_loader


# Creating the customized model, by adding a drop out and a dense layer on top of roberta to get the final output for the model.

class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.model_layer = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        output_1 = self.model_layer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output



def return_model(device):
    model = ModelClass()
    model = model.to(device)
    return model


# Function to calcuate the accuracy of the model

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


# Function to fine tune the model based on the epochs, model, tokenizer and other arguments

def train(epoch, model, device, training_loader, optimizer, loss_function):
    n_correct = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_loss = 0
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask).squeeze()
        optimizer.zero_grad()
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            wandb.log({"Training Loss per 100 steps": loss_step})
            wandb.log({"Training Accuracy per 100 steps": accu_step})

        optimizer.zero_grad()
        loss.backward()


        xm.optimizer_step(optimizer)
        xm.mark_step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    wandb.log({"Training Loss Epoch": epoch_loss})
    wandb.log({"Training Accuracy Epoch": epoch_accu})

# If you want to use GPU, change
# xm.optimizer_step(optimizer)  => optimizer.step()
# xm.mark_step()



def valid(epoch, model, device, validation_loader, loss_function):
    n_correct = 0;
    total = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_loss = 0
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 100 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                wandb.log({"Validation Loss per 100 steps": loss_step})
                wandb.log({"Validation Accuracy per 100 steps": accu_step})

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    wandb.log({"Validation Loss Epoch": epoch_loss})
    wandb.log({"Validation Accuracy Epoch": epoch_accu})
    print(f'The Validation Accuracy: {(n_correct * 100) / nb_tr_examples}')


def run():
    # WandB – Initialize a new run
    wandb.init(project="transformers_turkish_movie_sentiment")

    config = wandb.config
    config.MAX_LEN = 512
    config.TRAIN_BATCH_SIZE = 4
    config.VALID_BATCH_SIZE = 2
    config.EPOCHS = 4
    config.LEARNING_RATE = 1e-05
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    df = data.copy()
    pre = Preprocess(df)
    encoding_dict, df = pre.processing()

    training_loader, validation_loader = return_dataloader(df, tokenizer, config.TRAIN_BATCH_SIZE,
                                                           config.VALID_BATCH_SIZE, config.MAX_LEN)

    model = return_model(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        train(epoch, model, device, training_loader, optimizer, loss_function)

    valid(epoch, model, device, validation_loader, loss_function)


run()