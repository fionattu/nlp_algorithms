import re
import random


class Config:
    src_data = '../dataset/MSRA/train.txt'
    char2tag = '../data/MSRA/char2tag.txt'
    train_inputs = '../data/MSRA/train_inputs.txt'
    train_labels = '../data/MSRA/train_labels.txt'
    valid_inputs = '../data/MSRA/valid_inputs.txt'
    valid_labels = '../data/MSRA/valid_labels.txt'
    train_percentage = 0.8


class MSRADataSplit:
    """
    Use BME to tag characters: nr/人名 ns/处所 nt/团体
    """
    def __init__(self, config):
        self.src_data = config.src_data
        self.char2tag = config.char2tag
        self.train_inputs = config.train_inputs
        self.train_labels = config.train_labels
        self.valid_inputs = config.valid_inputs
        self.valid_labels = config.valid_labels
        self.train_percentage = config.train_percentage

    def load_data(self):
        """
        Read source data and produce char2tag file
        """
        line_cnt = 0
        with open(self.char2tag, 'w') as out:
            with open(self.src_data, 'r') as f:
                for line in f:
                    words = line.strip().split()
                    for word in words:
                        w, label = word.split('/')
                        w = w.strip()
                        if label == 'o' and len(w) > 0:
                            for cha in w:
                                out.write("{}/{} ".format(cha, label))
                        else:
                            if len(w) == 1:
                                out.write("{}/B_{} ".format(w, label))
                            elif len(w) > 1:
                                for cha in w:
                                    if cha == w[0]:
                                        out.write("{}/B_{} ".format(cha, label))
                                    elif cha == w[-1]:
                                        out.write("{}/E_{} ".format(cha, label))
                                    else:
                                        out.write("{}/M_{} ".format(cha, label))
                    out.write('\n')
                    line_cnt += 1
                    if line_cnt % 1000 == 0:
                        print("finish processing line: {}".format(line_cnt))
            f.close()
        out.close()

    def __load_sentences_tags(self):
        """
        load word2tag file into tuples of (sentence, tag)
        """
        sentences, tags = [], []
        with open(self.char2tag, 'r') as data:
            for line in data:
                line = re.split('[，。；！：？、‘’“”]/[o]', line.strip())
                for sentence in line:
                    sen_input, sen_label = [], []
                    words = sentence.strip().split()
                    has_ner = False
                    alpha_digits_cache = ''
                    for w in words:
                        word, label = w.split('/')
                        if bool(re.search(r'[\da-zA-Z]', word)):
                            alpha_digits_cache += word
                        else:
                            if len(alpha_digits_cache) != 0:
                                sen_input.append('1')
                                sen_label.append('o')
                                alpha_digits_cache = ''
                            sen_input.append(word)
                            sen_label.append(label)
                        if not has_ner and label != 'o':
                            has_ner = True
                    if len(alpha_digits_cache) != 0:
                        sen_input.append('1')
                        sen_label.append('o')
                    # only train sentences with named entities
                    if has_ner:
                        sentences.append(sen_input)
                        tags.append(sen_label)

        return list(zip(sentences, tags))

    def split_data(self):
        """
        Split (sentence, tag) tuples into train/validation dataset
        """
        full_data = self.__load_sentences_tags()
        train_size = int(len(full_data) * self.train_percentage)
        train_index = random.sample(range(len(full_data)), train_size)
        x_train, x_valid = [], []
        for i, v in enumerate(full_data):
            if i in train_index:
                x_train.append(v)
            else:
                x_valid.append(v)

        with open(self.train_inputs, 'w') as out_t_x:
            with open(self.train_labels, 'w') as out_t_y:
                for x in x_train:
                    out_t_x.write(','.join(x[0]) + '\n')
                    out_t_y.write(','.join(x[1]) + '\n')
            out_t_y.close()
        out_t_x.close()

        with open(self.valid_inputs, 'w') as out_v_x:
            with open(self.valid_labels, 'w') as out_v_y:
                for x in x_valid:
                    out_v_x.write(','.join(x[0]) + '\n')
                    out_v_y.write(','.join(x[1]) + '\n')
            out_v_y.close()
        out_v_x.close()


if __name__ == '__main__':
    msra_data_split = MSRADataSplit(Config())

    # create char2tag for all models
    # msra_data_split.load_data()

    # split char2tag to train/validation
    msra_data_split.split_data()
