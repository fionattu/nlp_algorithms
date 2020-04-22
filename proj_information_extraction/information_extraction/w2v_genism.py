import os
import json
import re


def read_file(path):
    with open(path, 'r') as f:
        text = f.read()
    return text


def process_text(text):
    text = text.lower()
    text = re.sub('[^a-z^0-9^\u4e00-\u9fa5]', '', text)
    text = re.sub('[0-9]', '0', text)
    return text


sentences = []
json_data_path = "/Users/fiona/Desktop/Github/IE_week2/dl_information_extract/dataset/json/"
json_files = [os.path.join(json_data_path, file) for file in os.listdir(json_data_path)]
for file in json_files:
    text = read_file(file)
    contexts = json.loads(text)
    for context in contexts:
        text = process_text(context['text'])
        sentences.append(' '.join(text))
print(len(sentences), sentences[0])
print()