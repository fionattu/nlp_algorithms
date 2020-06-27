from torch.utils.data import DataLoader
from bert import Bert, Config
from bert_dataset import BertDataset
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


def retrieve_entity(tag_ids):
    result, entity = [], []
    for i in range(len(tag_ids)):
        tag = tag_ids[i]  # like 'B_ns'
        if tag[0] == 'B':
            entity = [tag[2:], i, i]  # like ('ns', 2, 2)
        elif tag[0] == 'M' and len(entity) > 0 and entity[0] == tag[2:]:
            entity[2] = i
        elif tag[0] == 'E' and len(entity) > 0 and entity[0] == tag[2:]:
            entity[2] = i
            result.append(entity)
            entity = []
        else:
            if len(entity) > 0:
                result.append(entity)
                entity = []
    return result


def get_metrics(pred_tags, true_tags):
    pred_tags = retrieve_entity(pred_tags)
    true_tags = retrieve_entity(true_tags)
    union = [i for i in pred_tags if i in true_tags]  # TP
    precision = 100.0 * len(union) / len(pred_tags) if len(pred_tags) != 0 else 0.0
    recall = 100.0 * len(union) / len(true_tags) if len(true_tags) != 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if len(union) != 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1_score}


def evaluate(config, model):
    model.eval()
    valid_iter = DataLoader(BertDataset(config, config.valid),
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for input_ids, tag_ids, att_masks in valid_iter:
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
    return get_metrics(pred_tags, true_tags)


def train(config, model):
    # only fine-tune the linear classifier layer
    optimizer = optim.Adam(list(model.model.classifier.parameters()), config.lr)
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
            metrics = evaluate(config, model)
            logging.info('Epoch [{}/{}] precision: {}, recall:{}, f1: {}'.format(epoch + 1,
                                                                                 config.n_epochs,
                                                                                 metrics['precision'],
                                                                                 metrics['recall'],
                                                                                 metrics['f1']))
            if epoch > 10 and (abs(metrics['f1'] - f1) < config.f1_conv or metrics['f1'] < f1):
                break
            if metrics['f1'] > f1:
                f1 = metrics['f1']

        torch.save(model.state_dict(), config.model_save_path)


if __name__ == '__main__':
    set_seed()
    config = Config()
    model = Bert(config)
    train(config, model)
