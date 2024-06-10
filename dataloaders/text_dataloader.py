from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import random

class Text_DataLoader(Dataset):
    def __init__(self, queryfile_path, ):
        self.sentence_ids = []
        self.sentences = []
        with open(queryfile_path, 'r') as f:
            for line in f.readlines():
                sentence_id, sentence = line.strip().split('\t', 1)
                self.sentence_ids.append(sentence_id)
                self.sentences.append(sentence)
        self.sentence_num = len(self.sentence_ids)
        print("Sentence number: {}".format(self.sentence_num))

    def __len__(self):
        return self.sentence_num

    def __getitem__(self, idx):
        sentence_id = self.sentence_ids[idx]
        sentence = self.sentences[idx]

        return sentence_id, sentence