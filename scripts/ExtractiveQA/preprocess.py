import torch
from torch.utils.data import Dataset

class ExtractiveQADataset(Dataset):

    def __init__(self, datas, tokenizer):
        self.tokenizer = tokenizer
        self.length = len(datas)
        self.tokenized = self.preprocess(datas)

    def __len__(self):
        return self.length

    def tokenize(self, labels, texts):
        tokenized = self.tokenizer(
            labels,
            texts,
            padding="max_length",
            max_length=512,
            truncation="only_second",
            return_offsets_mapping=True
        )
        return tokenized        

    def preprocess(self, datas):
        labels = [data["label"].strip() for data in datas]
        texts = [data["text"].strip() for data in datas]
        targets = [data["extracted_part"] for data in datas]

        tokenized = self.tokenize(labels, texts)

        offset_mapping = tokenized.pop("offset_mapping")
        start_ids = []
        end_ids = []

        for i, offset in enumerate(offset_mapping):
          
            start_char = targets[i]["answer_start"][0]
            end_char = targets[i]["answer_end"][0]
            sequence_ids = tokenized.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            text_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            text_end = idx - 1
            
            # assume that if there is no answer for the context, start and end ids are [SEP] tokens,
            # that is why we append index 0 ([SEP] token is always the first token)
            if start_char == 0 and end_char == 0:
                start_ids.append(0)
                end_ids.append(0)

            else:
                idx = text_start
                while idx <= text_end and offset[idx][0] <= start_char:
                    idx += 1
                start_ids.append(idx - 1)

                idx = text_end
                while idx >= text_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_ids.append(idx + 1)

            tokenized["start_ids"] = start_ids
            tokenized["end_ids"] = end_ids
        
        return tokenized          

    def __getitem__(self, idx):
        inputs = {
            "input_ids": torch.LongTensor(self.tokenized["input_ids"][idx]),
            "attention_mask": torch.LongTensor(self.tokenized["attention_mask"][idx]),
            "start_ids": self.tokenized["start_ids"][idx],
            "end_ids": self.tokenized["end_ids"][idx]
        }
        return inputs