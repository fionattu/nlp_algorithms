import os
import sys
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent) + '/')

from torch.utils.data import DataLoader
from bert import Bert, Config
from bert_dataset import BertDataset
from utils.metrics import *
from itertools import accumulate
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
    if config.inference:
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
        valid_iter = DataLoader(BertDataset(config, config.infer),
                                batch_size=config.batch_size_infer,
                                shuffle=False,
                                num_workers=config.num_workers)
    else:
        # checkpoint = torch.load(config.model_save_path)
        # model.load_state_dict(checkpoint)
        valid_iter = DataLoader(BertDataset(config, config.valid),
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers)

    if config.use_gpu:
        model.cuda()
    model.eval()

    true_tags, pred_tags = [], []
    inputs_infer, masks_total = [], []
    with torch.no_grad():
        for input_ids, tag_ids, att_masks in valid_iter:
            if config.inference:
                masks_infer = torch.sum(att_masks, dim=1)
                masks_total += [i for i in masks_infer.numpy()]
                inputs_infer += [config.tokenizer.decode(input_ids[i][:masks_infer[i]].numpy())
                                 for i in range(input_ids.shape[0])]
            if config.use_gpu:
                input_ids = input_ids.to(config.device)
                tag_ids = tag_ids.to(config.device)
                att_masks = att_masks.to(config.device)
            _, outputs = model(input_ids, tag_ids, att_masks)  # outputs = [batch_size, seq_len, n_tags]

            outputs = outputs.detach().cpu().numpy()  # [batch_size, seq_len, n_tags]
            tags = tag_ids.to('cpu').numpy()  # [batch_size, seq_len]
            att_masks = att_masks.to('cpu').numpy().sum(1)  # [batch_size]
            pred = [config.id2tag[index] for i, indices in enumerate(np.argmax(outputs, axis=2)) for index in
                    indices[:att_masks[i]]]
            true = [config.id2tag[index] for i, indices in enumerate(tags) for index in indices[:att_masks[i]]]
            assert len(pred) == len(true)
            pred_tags.extend(pred)
            true_tags.extend(true)
    inputs_tags = ['{}/{}'.format(item, pred_tags[i]) for i, item in enumerate(' '.join(inputs_infer).split())]
    inputs_tags = [inputs_tags[x - y: x] for x, y in zip(accumulate(masks_total), masks_total)]
    return get_metrics(pred_tags, true_tags), inputs_tags


def train(config, model):
    if config.use_gpu:
        model.cuda()

    if config.fine_tune:
        # Fine-tune bert
        logging.info('Fine-tune bert and classifier params!')
        optimizer = optim.Adam(model.parameters(), config.bert_lr)
    else:
        # Feature-based bert
        logging.info('Fine-tune simply the classifier params!')
        optimizer = optim.Adam(list(model.model.classifier.parameters()), config.classifier_lr)

    f1 = -1000

    for epoch in range(config.n_epochs):
        model.train()
        logging.info('Epoch [{}/{}] starts'.format(epoch + 1, config.n_epochs))
        train_iter = DataLoader(BertDataset(config, config.train),
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.num_workers)

        for input_ids, tag_ids, att_masks in train_iter:

            if config.use_gpu:
                input_ids = input_ids.to(config.device)
                tag_ids = tag_ids.to(config.device)
                att_masks = att_masks.to(config.device)

            optimizer.zero_grad()
            loss, outputs = model(input_ids, tag_ids, att_masks)
            loss.backward()
            optimizer.step()
        logging.info('Epoch [{}/{}] loss: {}'.format(epoch + 1, config.n_epochs, loss.item()))

        if epoch % config.eval_freq == 0:
            metrics, _ = evaluate(config, model)
            logging.info('Epoch [{}/{}] precision: {}, recall:{}, f1: {}'.format(epoch + 1,
                                                                                 config.n_epochs,
                                                                                 metrics['precision'],
                                                                                 metrics['recall'],
                                                                                 metrics['f1']))
            if epoch > 2000 and (abs(metrics['f1'] - f1) < config.f1_conv or metrics['f1'] < f1):
                break
            if metrics['f1'] > f1:
                f1 = metrics['f1']

        torch.save(model.state_dict(), config.model_save_path)


def inference(config, model):
    res, inputs_tags = evaluate(config, model)
    print(res)
    for i in inputs_tags:
        print(i)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    torch.set_num_threads(10)
    set_seed()
    config = Config()
    model = Bert(config)

    if config.inference:
        inference(config, model)
    else:
        train(config, model)
