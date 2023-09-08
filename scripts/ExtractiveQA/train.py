from tqdm import tqdm
import json
import numpy as np
import torch
import torch.nn as nn

class ExtractiveQAModel(nn.Module):
    
    def __init__(self, pretrained_model, num_classes, dropout=0):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.pretrained_model.config.hidden_size, num_classes)
        self.pretrained_model.init_weights()
        
    def forward(self, input_ids, mask):
        outputs = self.pretrained_model(
            input_ids,
            attention_mask=mask,
        )
        sequence_output = outputs[0]
        logits = self.linear(self.dropout(sequence_output))
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return start_logits, end_logits


def train_iter(model, iterator, optimizer, criterion):

    epoch_loss = []
    epoch_start_accuracy = []
    epoch_end_accuracy = []
    epoch_accuracy = []

    model.train() 

    for x in tqdm(iterator):

        input_ids = x["input_ids"].to(DEVICE)
        mask = x["attention_mask"].to(DEVICE)
        start_ids = x["start_ids"].to(DEVICE)
        end_ids = x["end_ids"].to(DEVICE)

        optimizer.zero_grad()

        start_logits, end_logits = model(input_ids, mask)
        start_logits = start_logits.to(DEVICE)
        end_logits = end_logits.to(DEVICE)

        loss = None
        start_loss = criterion(start_logits, start_ids)
        end_loss = criterion(end_logits, end_ids)
        loss = (start_loss + end_loss) / 2

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        # compute metrics
        start_pred = start_logits.argmax(dim=1)
        end_pred = end_logits.argmax(dim=1)
        epoch_start_accuracy.append(float(torch.sum(start_pred == start_ids) / start_ids.size(0)))
        epoch_end_accuracy.append(float(torch.sum(end_pred == end_ids) / end_ids.size(0)))

        # accuracy
        start_pred = start_pred.to("cpu").numpy()
        end_pred = end_pred.to("cpu").numpy()
        start_ids = start_ids.to("cpu").numpy()
        end_ids = end_ids.to("cpu").numpy()
        accuracy_score = 0
        for i in np.stack((start_ids, end_ids), 1) == np.stack((start_pred, end_pred), 1):
            if np.all(i):
                accuracy_score += 1
        epoch_accuracy.append(float(accuracy_score / start_ids.size))

    return np.mean(epoch_loss), np.mean(epoch_accuracy), np.mean(epoch_start_accuracy), np.mean(epoch_end_accuracy)

def evaluate_iter(model, iterator, criterion):

    epoch_loss = []
    epoch_start_accuracy = []
    epoch_end_accuracy = []
    epoch_accuracy = []

    model.eval()  

    with torch.no_grad():

        for x in tqdm(iterator):

            input_ids = x["input_ids"].to(DEVICE)
            mask = x["attention_mask"].to(DEVICE)
            start_ids = x["start_ids"].to(DEVICE)
            end_ids = x["end_ids"].to(DEVICE)

            start_logits, end_logits = model(input_ids, mask)
            start_logits = start_logits.to(DEVICE)
            end_logits = end_logits.to(DEVICE)

            loss = None
            start_loss = criterion(start_logits, start_ids)
            end_loss = criterion(end_logits, end_ids)
            loss = (start_loss + end_loss) / 2

            epoch_loss.append(loss.item())

            # compute metrics
            start_pred = start_logits.argmax(dim=1)
            end_pred = end_logits.argmax(dim=1)
            epoch_start_accuracy.append(float(torch.sum(start_pred == start_ids) / start_ids.size(0)))
            epoch_end_accuracy.append(float(torch.sum(end_pred == end_ids) / end_ids.size(0)))

            # accuracy
            start_pred = start_pred.to("cpu").numpy()
            end_pred = end_pred.to("cpu").numpy()
            start_ids = start_ids.to("cpu").numpy()
            end_ids = end_ids.to("cpu").numpy()
            accuracy_score = 0
            for i in np.stack((start_ids, end_ids), 1) == np.stack((start_pred, end_pred), 1):
                if np.all(i):
                    accuracy_score += 1
            epoch_accuracy.append(float(accuracy_score / start_ids.size))

    return np.mean(epoch_loss), np.mean(epoch_accuracy), np.mean(epoch_start_accuracy), np.mean(epoch_end_accuracy)

def train_epochs(epochs, model, train_loader, valid_loader, optimizer, scheduler, criterion, mode):

    train_losses = []
    train_accuracy_scores = []
    train_start_scores = []
    train_end_scores = []

    valid_losses = []
    valid_accuracy_scores = []
    valid_start_scores = []
    valid_end_scores = []

    best_valid_loss = 1e+6

    for i in range(epochs):

        print(f"\nEpoch: {i+1}")

        train_loss, train_accuracy, train_start_score, train_end_score = train_iter(model, train_loader, optimizer, criterion)
        train_losses.append(float(train_loss))
        train_accuracy_scores.append(float(train_accuracy))
        train_start_scores.append(float(train_start_score))
        train_end_scores.append(float(train_end_score))
        print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}, Train start score: {train_start_score}, Train end score: {train_end_score}")

        val_loss, val_accuracy, val_start_score, val_end_score = evaluate_iter(model, valid_loader, criterion)
        valid_losses.append(float(val_loss))
        valid_accuracy_scores.append(float(val_accuracy))
        valid_start_scores.append(float(val_start_score))
        valid_end_scores.append(float(val_end_score))
        print(f"Valid loss: {val_loss}, Valid accuracy: {val_accuracy}, Valid start score: {val_start_score}, Valid end score: {val_end_score}")

        scheduler.step(val_loss)
        
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), f"models/ExtractiveQA/{mode}/state_dict_model.pth")
            torch.save(optimizer.state_dict(), f"models/ExtractiveQA/{mode}/state_dict_optimizer.pth")
            torch.save(scheduler.state_dict(), f"models/ExtractiveQA/{mode}/state_dict_scheduler.pth")

        with open(f'models/ExtractiveQA/{mode}/info.json', 'w') as file_object:
                    info = {
                        'train_losses': train_losses,
                        'train_accuracy': train_accuracy_scores,
                        'train_start_evals': train_start_scores,
                        'train_end_evals': train_end_scores,
                        'valid_losses': valid_losses,
                        'valid_accuracy': valid_accuracy_scores,
                        'valid_start_evals': valid_start_scores,
                        'valid_end_evals': valid_end_scores
                    }
                    file_object.write(json.dumps(info, indent=2))