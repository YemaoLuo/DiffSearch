import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embedding(sentence):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(sentence_embeddings, p=2, dim=1)


def get_similarity(sentence1, sentence2):
    return np.inner(sentence1, sentence2)[0][0]


if __name__ == '__main__':
    message = ''
    while message != 'quit':
        message = input('Press Enter a comment message...\n')
        flag = input('Press Enter 0(true) or 1(false) to return diff...\n')
        k = eval(input('Press Enter a result size...\n'))
        message_embedding = get_embedding(message)
        tensors = os.listdir('./log/tensor')
        tensor_list = []
        pbar = tqdm(total=len(tensors) - 1, desc='Processing', unit='items', position=0, leave=True)
        for tensor in tensors:
            pbar.update(1)
            tensor_map = {'id': tensor}
            value = torch.load('./log/tensor/' + tensor)
            tensor_map['tensor'] = value
            tensor_map['similarity'] = get_similarity(message_embedding, value)
            tensor_list.append(tensor_map)
        sorted_data = sorted(tensor_list, key=lambda x: x['similarity'], reverse=True)
        print('*' * 50)
        for i in range(k):
            id = sorted_data[i]['id']
            message_file = open('./log/' + id + '.log', 'r')
            lines = message_file.readlines()
            print('id:', id, '\nauthor:', lines[4].replace('\n', ''), '\ntime:', lines[6].replace('\n', '')
                  , '\nmessage:', lines[2].replace('\n', ''), '\nsimilarity:', sorted_data[i]['similarity'])
            if flag == '0':
                print('diff:')
                diff = ''
                for i in range(7, len(lines)):
                    diff += lines[i]
                print(diff)
            print('*' * 50)
