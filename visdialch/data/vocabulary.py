import json
import os
from typing import List


class Vocabulary(object):

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<S>"
    EOS_TOKEN = "</S>"
    UNK_TOKEN = "<UNK>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, word_counts_path: str, min_count: int = 5):
        if not os.path.exists(word_counts_path):
            raise FileNotFoundError(f"Word counts do not exist at {word_counts_path}")

        with open(word_counts_path, "r") as word_counts_file:
            word_counts = json.load(word_counts_file)

            word_counts = [
                (word, count) for word, count in word_counts.items() if count >= min_count
            ]
            word_counts = sorted(word_counts, key=lambda wc: -wc[1])
            words = [w[0] for w in word_counts]

        self.word2index = {}
        self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
        self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
        self.word2index[self.EOS_TOKEN] = self.EOS_INDEX
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
        for index, word in enumerate(words):
            self.word2index[word] = index + 4

        self.index2word = {index: word for word, index in self.word2index.items()}

    @classmethod
    def from_saved(cls, saved_vocabulary_path: str) -> "Vocabulary":
        with open(saved_vocabulary_path, "r") as saved_vocabulary_file:
            cls.word2index = json.load(saved_vocabulary_file)
        cls.index2word = {index: word for word, index in cls.word2index.items()}

    def to_indices(self, words: List[str]) -> List[int]:
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def to_words(self, indices: List[int]) -> List[str]:
        return [self.index2word.get(index, self.UNK_TOKEN) for index in indices]

    def save(self, save_vocabulary_path: str) -> None:
        with open(save_vocabulary_path, "w") as save_vocabulary_file:
            json.dump(self.word2index, saved_vocabulary_file)

    def __len__(self):
        return len(self.index2word)

