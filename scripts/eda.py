from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher

import seaborn as sns
import matplotlib.pyplot as plt

class ExploratoryDataAnalysis:

    def sanity_check(self, datas):

        # check if `texts`, `answer_start` and `answer_end` in `extracted parts` contain only one element each
        parts_counts = {
            "text": 0,
            "answer_start": 0,
            "answer_end": 0
        }
        for data in datas:
            for part in parts_counts:
                if len(data["extracted_part"][part]) > 1:
                    parts_counts[part] += 1
        for part, count in parts_counts.items():
            print(f"Count of `{part}` that have more than one element: {count}")

        # check if indices in `answer_start` - `answer_end` and indices of extracted span in text are the same
        difference = 0
        for data in datas:
            start_idx = data["extracted_part"]["answer_start"][0]
            end_idx = data["extracted_part"]["answer_end"][0]
            if data["text"][start_idx:end_idx] != data["extracted_part"]["text"][0]:
                difference += 1
        print(f"Count of examples where `answer_start` and `answer_end` are not correct: {difference}")

    def explore_labels_balance(self, datas):

        state = f"Data Size: {len(datas)}"
        print(f"{state}\n{'=' * len(state)}")

        state = "Labels Balance"
        print(f"\n\n{state}\n{'=' * len(state)}")
        labels_proportion = Counter()
        for data in datas:
            labels_proportion[data["label"]] += 1 / len(datas)
        for label, value in labels_proportion.items():
            print(f"\t- {label}: {round(value, 2)}")

    def explore_texts_lengths(self, texts):
      
        state = "Texts Lengths"
        print(f"{state}\n{'=' * len(state)}")

        def print_info(lengths):
            print("\t\tAverage length:\t\t\t", int(np.mean(lengths)))
            print("\t\tMedian length:\t\t\t", int(np.median(lengths)))
            print("\t\tMinimum length:\t\t\t", min(lengths))
            print("\t\tMaximum length:\t\t\t", max(lengths))
            print("\t\t90-percentile:\t\t\t", int(np.percentile(lengths, 90)))
            print("\t\t99-percentile:\t\t\t", int(np.percentile(lengths, 99)))
            print("\t\tMin length texts (count):\t", lengths.count(min(lengths)))

        lengths_by_char = [len(text) for text in texts]
        print("\tLengths: symbols")
        print_info(lengths_by_char)

        lengths_by_token = [len(text.split()) for text in texts]
        print("\tLengths: base tokens received by `.split()` method")
        print_info(lengths_by_token)

        df = pd.DataFrame({"Symbols": lengths_by_char, "Tokens": lengths_by_token})

        fig, axes = plt.subplots(1, 2, figsize=(21, 3))
        plt.suptitle('Length Distribution')
        for feature, ax in zip(["Symbols", "Tokens"], axes.flatten()):
            sns.histplot(df[feature], ax=ax, color='#7eb19c', kde=True)

def collect_similarities(output_file: str, texts: list):

    def similar(text_1, text_2):
        return SequenceMatcher(None, text_1, text_2).ratio()

    similarities = {}
    for i in tqdm(range(len(texts) - 1)):
        for j in range(i + 1, len(texts)):
            similarities[(i, j)] = similar(texts[i], texts[j])

    output_f = open(output_file, "w", encoding="utf-8")
    output_f.write("text_1,text_2,sim_score\n")
    for texts, sim in similarities.items():
        output_f.write(f"{texts[0]},{texts[1]},{sim}\n")

def get_similar_texts(similarities_df, train_datas):

    df = similarities_df[similarities_df.sim_score > 0.9]
    df = df.sort_values(['text_1', 'text_2'])

    # get groups of similar documents
    groups = []
    for idx, row in df.iterrows():
        idx_1 = int(row["text_1"])
        idx_2 = int(row["text_2"])
        if groups:
            count = 0
            for group in groups:
                if (idx_1 in group) and (idx_2 not in group):
                    group.append(idx_2)
                    count += 1
                elif (idx_2 in group) and (idx_1 not in group):      
                    group.append(idx_1)
                    count += 1
                elif (idx_1 in group) and (idx_2 in group):
                    count += 1
            if not count:
                groups.append([idx_1, idx_2])
        else:
            groups.append([idx_1, idx_2])

    # it migth be that one value is in two groups 
    # example:
    # first and second sim_score = 0.9, second and third sim_score = 0.9, and first and third sim_score is between 0.81 and 1)
    groups_flat_list = [idx for group in groups for idx in group]
    double_ids = [item for item, count in Counter(groups_flat_list).items() if count > 1]
    for double_idx in double_ids:
        group_to_join = []
        for group in groups:
            if double_idx in group:
                group_to_join.extend(group)
        groups = [group for group in groups if double_idx not in group]
        groups += [list(set(group_to_join))]
        groups_flat_list = [idx for group in groups for idx in group]
        double_ids = [item for item, count in Counter(groups_flat_list).items() if count > 1]

    # we do not delete similar entries which have different labels or different extracted texts
    groups_info = {}
    n = 0
    for group in groups:
        existing = []
        ids_to_delete = []
        for idx in group:
            idx = int(idx)
            label = train_datas[idx]["label"]
            extracted_text = train_datas[idx]["extracted_part"]["text"][0]
            if (label, extracted_text) not in existing:
                existing.append((label, extracted_text))
            else:
                ids_to_delete.append(idx)
        groups_info[n] = {
            "initial_count": len(group),
            "deleted_count": len(ids_to_delete),
            "deleted_ids": ids_to_delete
        }
        n += 1

    groups_info_df = pd.DataFrame.from_dict(groups_info, orient="index")
    deleted_ids_list = []
    for ids_list in groups_info_df.deleted_ids.values:
        deleted_ids_list.extend(ids_list)

    return deleted_ids_list





