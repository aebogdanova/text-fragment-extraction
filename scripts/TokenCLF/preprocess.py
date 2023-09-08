import torch
from torch.utils.data import Dataset

LABEL2ID = {
    "NO-LABEL": 0,
    "обеспечение исполнения контракта": 1, 
    "обеспечение гарантийных обязательств": 2
}

class TokenCLFDataset(Dataset):

    def __init__(self, tokenizer, datas):
        self.tokenizer = tokenizer
        self.length = len(datas)
        self.preprocessed = self.preprocess(datas)

    def __len__(self):
        return self.length

    def tokenize(self, texts):
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        return tokenized

    def preprocess(self, datas):
        queries_seq = [data["label"].strip() for data in datas]
        texts_seq = [data["text"].strip() for data in datas]
        targets_seq = [data["extracted_part"] for data in datas]

        tokens_seq = self.tokenize(texts_seq)
        offset_mapping = tokens_seq.pop("offset_mapping")
        labels_seq = []

        for offset_idx, offset in enumerate(offset_mapping):
            start_char = targets_seq[offset_idx]["answer_start"][0]
            end_char = targets_seq[offset_idx]["answer_end"][0]
            query = queries_seq[offset_idx]
            sequence_ids = tokens_seq.sequence_ids(offset_idx)
            
            # Find the start and end of the text
            text_start = 1
            idx = 1
            while sequence_ids[idx] == 0:
                idx += 1
            text_end = idx - 1

            # Find the indices of start and end tokens
            idx = text_start
            while idx <= text_end and offset[idx][0] <= start_char:
                idx += 1
            start_idx = idx - 1
            idx = text_end
            while idx >= text_start and offset[idx][1] >= end_char:
                idx -= 1
            end_idx = idx + 1
            
            labels = torch.zeros(512)
            for i, sequence_idx in enumerate(sequence_ids):
                if sequence_idx == None:
                    labels[i] = -100
                elif i >= start_idx and i <= end_idx:
                    labels[i] = LABEL2ID[query]
                else:
                    labels[i] = LABEL2ID["NO-LABEL"]
            labels_seq.append(labels.long())

            tokens_seq["labels"] = labels_seq

        return tokens_seq

    def __getitem__(self, idx):
        inputs = {
            "input_ids": torch.LongTensor(self.preprocessed["input_ids"][idx]),
            "attention_mask": torch.LongTensor(self.preprocessed["attention_mask"][idx]),
            "labels": self.preprocessed["labels"][idx]
        }
        return inputs