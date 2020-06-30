import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.getcwd()).parent) + '/')

from torch.utils.data import DataLoader
from bert_bilstm_crf import Bert_BiLSTM_CRF, Config
from bert_model.bert_dataset import BertDataset, collate_fn
from utils.metrics import *
import torch.optim as optim
import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO)


def set_seed():
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(config, model):
    model.eval()
    valid_iter = DataLoader(BertDataset(config, config.valid),
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers,
                            collate_fn=collate_fn)
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for input_ids, tag_ids, att_masks, lengths in valid_iter:
            if config.use_gpu:
                input_ids = input_ids.to(config.device)
                tag_ids = tag_ids.to(config.device)
                att_masks = att_masks.to(config.device)
                lengths = lengths.to(config.device)

            _, outputs = model(input_ids, att_masks, lengths)  # outputs = [batch_size, seq_len, n_valid_tags]

            tags = tag_ids.to('cpu').numpy()  # [batch_size, seq_len]
            att_masks = att_masks.to('cpu').numpy().sum(1)  # [batch_size]
            pred = [config.id2tag[index] for indices in outputs for index in indices]
            true = [config.id2tag[index] for i, indices in enumerate(tags) for index in indices[:att_masks[i]]]
            assert len(pred) == len(true)

            pred_tags.extend(pred)
            true_tags.extend(true)
    return get_metrics(pred_tags, true_tags)


def train(config, model):
    if config.use_gpu:
        model.cuda()

    if config.fine_tune:
        # Fine-tune bert
        logging.info('Fine-tune bert and bilstm-crf params!')
        optimizer = optim.Adam(model.parameters(), config.bert_lr)
    else:
        # Feature-based bert
        logging.info('Fine-tune simply the bilstm-crf params!')
        optimizer = optim.Adam(list(model.bilstm_crf.parameters()), config.bilstm_crf_lr)
    f1 = -1000

    for epoch in range(config.n_epochs):
        model.train()
        logging.info('Epoch [{}/{}] starts'.format(epoch + 1, config.n_epochs))
        train_iter = DataLoader(BertDataset(config, config.train),
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers,
                                collate_fn=collate_fn)

        for input_ids, tag_ids, att_masks, lengths in train_iter:
            if config.use_gpu:
                input_ids = input_ids.to(config.device)
                tag_ids = tag_ids.to(config.device)
                att_masks = att_masks.to(config.device)
                lengths = lengths.to(config.device)

            optimizer.zero_grad()

            loss = model.nll(input_ids, att_masks, tag_ids, lengths)

            loss.backward()
            optimizer.step()
        logging.info('Epoch [{}/{}] loss: {}'.format(epoch + 1, config.n_epochs, loss.item()))

        if epoch % config.eval_freq == 0:
            metrics = evaluate(config, model)
            logging.info('Epoch [{}/{}] precision: {}, recall:{}, f1: {}'.format(epoch + 1,
                                                                                 config.n_epochs,
                                                                                 metrics['precision'],
                                                                                 metrics['recall'],
                                                                                 metrics['f1']))
            if epoch > 10000 and (abs(metrics['f1'] - f1) < config.f1_conv or metrics['f1'] < f1):
                break
            if metrics['f1'] > f1:
                f1 = metrics['f1']

        torch.save(model.state_dict(), config.model_save_path)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    torch.set_num_threads(10)

    set_seed()
    config = Config()
    model = Bert_BiLSTM_CRF(config)
    train(config, model)
