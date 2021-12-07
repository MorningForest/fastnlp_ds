import sys
sys.path.append("../")

from datasets import load_dataset,  load_metric
from datasets import DatasetDict, set_caching_enabled
from pprint import pprint

from transformers import BertTokenizerFast, BertModel, DataCollatorWithPadding
from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

##使用 datasets写一个demo
##关闭cache
set_caching_enabled(False)


def process_data(ds: DatasetDict, tokenizer: BertTokenizerFast, max_len=256)-> DatasetDict:
    ## 分词
    ds = ds.map(lambda x: tokenizer(x['text'], max_length=max_len), batched=True,
                remove_columns=['text'])
    ds.set_format(type='torch', columns=ds.column_names)
    return ds

class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, 200),
            nn.ReLU(),
            nn.Linear(200, 2)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_cls = self.bert(input_ids, token_type_ids, attention_mask)[0][:, 0]
        logit = self.classifier(bert_cls)
        return logit

def trainer(opt, loss_fn, train_dl, val_dl, epoch, model, metric, device):

    for ep in range(epoch):
        total_loss, total_acc, num = 0, 0, 0
        for batch in tqdm(train_dl, desc='train epoch'):
            opt.zero_grad()
            y_true = batch['labels'].to(device)
            inp = {
                'input_ids': batch['input_ids'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            y_pred = model(**inp)
            loss = loss_fn(y_pred, y_true)

            total_acc += metric.compute(references=y_true.cpu().numpy(), predictions=y_pred.cpu().argmax(dim=-1).numpy())['accuracy']
            total_loss += loss.item()
            num += 1
            loss.backward()
            opt.step()
        total_acc /= num
        total_loss /= num
        model.eval()
        val_loss, val_acc, val_num = 0, 0, 0
        for batch in tqdm(val_dl, desc='val epoch'):
            opt.zero_grad()
            y_true = batch['label']
            inp = {
                'input_ids': batch['input_ids'],
                'token_type_ids': batch['token_type_ids'],
                'attention_mask': batch['attention_mask']
            }
            y_pred = model(**inp)
            val_loss += loss_fn(y_pred, y_true).item()
            total_acc += metric.compute(references=y_true.numpy(), predictions=y_pred.argmax(dim=-1).numpy())['accuracy']
            val_num += 1
        val_loss /= val_num
        val_acc /= val_num
        pprint("epoch: {}, loss: {}, acc: {} val_loss: {} val_acc: {}".format(ep, total_loss, total_acc, val_loss, val_acc))

def run():
    imdb_train = load_dataset('imdb', split='train').shuffle(seed=123)
    imdb_test = load_dataset('imdb', split='test').shuffle(seed=321)
    imdb_split = imdb_train.train_test_split(test_size=0.2, seed=123)
    imdb_train = imdb_split['train']
    imdb_val = imdb_split['test']

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    imdb_train = process_data(ds=imdb_train, tokenizer=tokenizer).rename_column('label', 'labels')
    imdb_val = process_data(ds=imdb_val, tokenizer=tokenizer).rename_column('label', 'labels')
    imdb_test = process_data(ds=imdb_test, tokenizer=tokenizer).rename_column('label', 'labels')

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dl = DataLoader(imdb_train, batch_size=16, collate_fn=data_collator)
    val_dl = DataLoader(imdb_val, batch_size=16, collate_fn=data_collator)
    test_dl = DataLoader(imdb_test, batch_size=16, collate_fn=data_collator)

    model = ImdbModel().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()
    metric = load_metric('accuracy')
    trainer(opt, loss_fn=loss, train_dl=train_dl, val_dl=val_dl, epoch=10, model=model, metric=metric, device=device)
