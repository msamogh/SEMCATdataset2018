import os
import pickle

import gensim.models

import torch.utils.data
import trustfall


class SEMCATDataset(torch.utils.data.Dataset):

    # Mapping from canonical category names to filenames
    WORDS_DIR = '../Categories'
    CATEGORY_FILES = {
        f.split('-')[0]: f for f in os.listdir(WORDS_DIR)
    }

    def __init__(self, transform=None):
        self.transform = transform
        self.words = []
        for category, words_file in SEMCATDataset.CATEGORY_FILES.iteritems():
            self.words.extend([{
                'data': word,
                'category': category
            } for word in self._get_words_for_category(words_file)])

    def _get_words_for_category(self, words_file):
        words_file = os.path.join(
            SEMCATDataset.WORDS_DIR, words_file)
        with open(words_file, 'r') as f:
            words = map(lambda x: x.strip(), f.readlines())
            return words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        sample = self.words[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class OneVsAll(object):

    def __init__(self, target_category):
        self.target_category = target_category

    def __call__(self, sample):
        return {
            'data': sample['data'],
            'category': int(sample['category'] == self.target_category)
        }


class Text8Embedding(object):

    def __init__(self, text8_model_path):
        self.model = pickle.load(open(text8_model_path, 'rb'))

    def __call__(self, sample):
        return {
            'data': self.model.wv[sample['data']],
            'category': sample['category']
        }
