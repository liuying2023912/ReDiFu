import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np


class IEMOCAPDataset(Dataset):
    def __init__(self, path, windows, train=True):
        self.videolDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        self.windows = windows

    def get_semantic_adj(self, data):
        semantic_adj = []
        max_len = max(len(d[3]) for d in data)
        batch_speakers = [d[3].tolist() for d in data]

        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0 or j >= len(speaker):
                        continue
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1 if i == j else 2 if i < j else 3
                    else:
                        s[i, j] = 4 if i < j else 5
            semantic_adj.append(s)
        return torch.stack(semantic_adj)

    def getSelf_semantic_adj(self, data):
        Self_semantic_adj = []
        max_len = max(len(d[3]) for d in data)
        batch_speakers = [d[3].tolist() for d in data]

        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0 or j >= len(speaker):
                        continue
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1 if i == j else 2 if i > j else 3
            Self_semantic_adj.append(s)
        return torch.stack(Self_semantic_adj)

    def getCross_semantic_adj(self, data):
        Cross_semantic_adj = []
        max_len = max(len(d[3]) for d in data)
        batch_speakers = [d[3].tolist() for d in data]

        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0 or j >= len(speaker):
                        continue
                    if speaker[i] != speaker[j]:
                        s[i, j] = 4 if i < j else 5
                    elif i == j:
                        s[i, j] = 1
            Cross_semantic_adj.append(s)
        return torch.stack(Cross_semantic_adj)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]]),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),
            torch.LongTensor(self.videoLabels[vid])
        )

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        Self_semantic_adj = self.getSelf_semantic_adj(data)
        Cross_semantic_adj = self.getCross_semantic_adj(data)
        Semantic_adj = self.get_semantic_adj(data)

        data = [
            pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist()
            for i in dat
        ]
        data.append(torch.LongTensor(Self_semantic_adj))
        data.append(torch.LongTensor(Cross_semantic_adj))
        data.append(torch.LongTensor(Semantic_adj))
        return data


class MELDDataset(Dataset):
    def __init__(self, path, windows, train=True):
        videoIDs, videoSpeakers, videoLabels, videoText, \
        roberta2, roberta3, roberta4, \
        videoAudio, videoVisual, videoSentence, trainVid, \
        testVid, _ = pickle.load(open(path, 'rb'), encoding='latin1')

        self.videoIDs = videoIDs
        self.videoSpeakers = videoSpeakers
        self.videoLabels = videoLabels
        self.videoText = videoText
        self.roberta2 = roberta2
        self.roberta3 = roberta3
        self.roberta4 = roberta4
        self.videoAudio = videoAudio
        self.videoVisual = videoVisual
        self.videoSentence = videoSentence
        self.trainVid = trainVid
        self.testVid = testVid
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.windows = windows
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return (
            torch.FloatTensor(np.array(self.videoText[vid])),
            torch.FloatTensor(np.array(self.videoVisual[vid])),
            torch.FloatTensor(np.array(self.videoAudio[vid])),
            torch.IntTensor(self.videoSpeakers[vid]),
            torch.FloatTensor([1] * len(self.videoLabels[vid])),
            torch.LongTensor(self.videoLabels[vid])
        )

    def __len__(self):
        return self.len

    def get_semantic_adj(self, data):
        semantic_adj = []
        max_len = max(len(d[3]) for d in data)
        batch_speakers = [d[3].tolist() for d in data]

        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1 if i == j else 2 if i < j else 3
                    else:
                        s[i, j] = 4 if i < j else 5
            semantic_adj.append(s)
        return torch.stack(semantic_adj)

    def getSelf_semantic_adj(self, data):
        Self_semantic_adj = []
        max_len = max(len(d[3]) for d in data)
        batch_speakers = [d[3].tolist() for d in data]

        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0 or j >= len(speaker):
                        continue
                    if speaker[i] == speaker[j]:
                        s[i, j] = 1 if i == j else 2 if i > j else 3
            Self_semantic_adj.append(s)
        return torch.stack(Self_semantic_adj)

    def getCross_semantic_adj(self, data):
        Cross_semantic_adj = []
        max_len = max(len(d[3]) for d in data)
        batch_speakers = [d[3].tolist() for d in data]

        for speaker in batch_speakers:
            s = torch.zeros(max_len, max_len, dtype=torch.long)
            for i in range(len(speaker)):
                for j in range(i - self.windows, i + self.windows + 1):
                    if j < 0 or j >= len(speaker):
                        continue
                    if speaker[i] != speaker[j]:
                        s[i, j] = 4 if i < j else 5
                    elif i == j:
                        s[i, j] = 1
            Cross_semantic_adj.append(s)
        return torch.stack(Cross_semantic_adj)

    def return_labels(self):
        return [label for key in self.keys for label in self.videoLabels[key]]

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        Self_semantic_adj = self.getSelf_semantic_adj(data)
        Cross_semantic_adj = self.getCross_semantic_adj(data)
        Semantic_adj = self.get_semantic_adj(data)

        data = [
            pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist()
            for i in dat
        ]
        data.append(torch.LongTensor(Self_semantic_adj))
        data.append(torch.LongTensor(Cross_semantic_adj))
        data.append(torch.LongTensor(Semantic_adj))
        return data
