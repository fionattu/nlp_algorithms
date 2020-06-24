from torch.utils.data import DataLoader
from bert import Bert, Config
from dataset import BertDataset
import torch.optim as optim
import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO)


def evaluate(config, model, valid_iter):
    model.eval()
    true_tags, pred_tags = [], []
    with torch.no_grad():
        for input_ids, tag_ids, att_masks in valid_iter:
            loss, outputs = model(input_ids, tag_ids, att_masks) # outputs = [batch_size, seq_len, n_tags]

            inputs = input_ids.detach().cpu().numpy() # [batch_size, seq_len]
            outputs = outputs.detach().cpu().numpy()  # [batch_size, seq_len, n_tags]
            tags = tag_ids.to('cpu').numpy()          # [batch_size, seq_len]
            att_masks = att_masks.to('cpu').numpy().sum(1)  # [batch_size]
            pred = [[indices[:att_masks[i]]] for i, indices in enumerate(np.argmax(outputs, axis=2))]
            true = [[indices[:att_masks[i]]] for i, indices in enumerate(tags)]
            print()
            print()


def train(config, model, train_iter, valid_iter):
    model.train()
    optimizer = optim.Adam(model.parameters(), config.lr)

    for epoch in range(config.n_epochs):

        logging.info('Epoch [{}/{}] starts'.format(epoch + 1, config.n_epochs))

        for input_ids, tag_ids, att_masks in train_iter:

            optimizer.zero_grad()

            if config.use_gpu:
                input_ids = input_ids.to(config.device)
                tag_ids = tag_ids.to(config.device)
                att_masks = att_masks.to(config.device)

            loss, outputs = model(input_ids, tag_ids, att_masks)
            outputs = outputs.detach().cpu().numpy()  # [batch_size, seq_len, n_tags]
            # tags = tag_ids.to('cpu').numpy()
            att_masks = att_masks.to('cpu').numpy().sum(1)
            # pred = [[indices] for indices in np.argmax(outputs, axis=2)]
            # true = [[indices] for indices in tag_ids]
            pred = [[indices[:att_masks[i]]] for i, indices in enumerate(np.argmax(outputs, axis=2))]
            print()

            loss.backward()

            optimizer.step()

        logging.info('Epoch [{}/{}] loss: {}'.format(epoch + 1, config.n_epochs, loss.item()))


if __name__ == '__main__':
    config = Config()
    train_iter = DataLoader(BertDataset(config, config.train),
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    valid_iter = DataLoader(BertDataset(config, config.valid),
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.num_workers)
    model = Bert(config)
    train(config, model, train_iter, valid_iter)
