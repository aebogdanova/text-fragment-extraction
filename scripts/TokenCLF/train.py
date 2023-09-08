from tqdm import tqdm
import json
import numpy as np
import torch
import torch.nn as nn

LABEL2ID = {
    "NO-LABEL": 0,
    "обеспечение исполнения контракта": 1, 
    "обеспечение гарантийных обязательств": 2
}

class TokenCLFModel(torch.nn.Module):
    
    def __init__(self, pretrained_model, droupout=0.5, num_classes=len(LABEL2ID)):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(droupout)
        self.linear = nn.Linear(self.pretrained_model.config.hidden_size, num_classes)
        self.pretrained_model.init_weights()
        
    def forward(self, input_ids, mask):
        outputs = self.pretrained_model(
            input_ids,
            attention_mask=mask,
        )
        sequence_output = outputs[0]
        logits = self.linear(self.dropout(sequence_output))
        return logits


def compute_accuracy(pred_labels, active_labels):

    pred = torch.where(active_labels != -100, pred_labels, 0)
    active = torch.where(active_labels != -100, active_labels, 0)

    accuracy_score = 0
    for i in range(active.size(0)):
        if torch.all(pred[i] == active[i]):
            accuracy_score += 1
    
    return accuracy_score / active.size(0)


def train_iter(model, iterator, optimizer, criterion):

    epoch_loss = []
    epoch_accuracy = []
    epoch_label_score = []

    model.train() 

    for x in tqdm(iterator):

        input_ids = x["input_ids"].to(DEVICE)
        mask = x["attention_mask"].to(DEVICE)
        labels = x["labels"].to(DEVICE)

        optimizer.zero_grad()

        logits = model(input_ids, mask)

        active_loss = mask.view(-1) == 1
        active_logits = logits.view(-1, len(LABEL2ID))
        active_labels = torch.where(
                    active_loss, 
                    labels.view(-1), 
                    torch.tensor(criterion.ignore_index).type_as(labels)
                )
        loss = criterion(active_logits, active_labels)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        labels_pred = active_logits.argmax(dim=1)
        epoch_label_score.append(float(torch.sum(labels_pred == active_labels) / active_labels.size(0)))

        # accuracy
        labels_pred = labels_pred.to("cpu").numpy()
        active_loss_for_accuracy = mask == 1
        active_labels_for_accuracy = torch.where(
                        active_loss_for_accuracy, 
                        labels, 
                        torch.tensor(criterion.ignore_index).type_as(labels)
                    ).to("cpu")
        pred_labels_for_accuracy = logits.argmax(2).to("cpu")
        epoch_accuracy.append(compute_accuracy(
            pred_labels_for_accuracy,
            active_labels_for_accuracy
        ))

      
    return np.mean(epoch_loss), np.mean(epoch_accuracy), np.mean(epoch_label_score)

def evaluate_iter(model, iterator, criterion):

    epoch_loss = []
    epoch_accuracy = []
    epoch_label_score = []

    model.eval()  

    with torch.no_grad():
        for x in tqdm(iterator):
            input_ids = x["input_ids"].to(DEVICE)
            mask = x["attention_mask"].to(DEVICE)
            labels = x["labels"].to(DEVICE)

            logits = model(input_ids, mask)

            active_loss = mask.view(-1) == 1
            active_logits = logits.view(-1, len(LABEL2ID))
            active_labels = torch.where(
                        active_loss, 
                        labels.view(-1), 
                        torch.tensor(criterion.ignore_index).type_as(labels)
                    )
            loss = criterion(active_logits, active_labels)

            epoch_loss.append(loss.item())

            labels_pred = active_logits.argmax(1).to("cpu")
            epoch_label_score.append(torch.sum(labels_pred == active_labels.to("cpu")) / active_labels.size(0))

            # accuracy
            active_loss_for_accuracy = mask == 1
            active_labels_for_accuracy = torch.where(
                            active_loss_for_accuracy, 
                            labels, 
                            torch.tensor(criterion.ignore_index).type_as(labels)
                        ).to("cpu")
            pred_labels_for_accuracy = logits.argmax(2).to("cpu")
            epoch_accuracy.append(compute_accuracy(
                pred_labels_for_accuracy,
                active_labels_for_accuracy
            ))

    return np.mean(epoch_loss), np.mean(epoch_accuracy), np.mean(epoch_label_score)

def train_epochs(epochs, model, train_loader, valid_loader, optimizer, scheduler, criterion):

    train_losses = []
    train_accuracy_scores = []
    train_labels_scores = []

    valid_losses = []
    valid_accuracy_scores = []
    valid_labels_scores = []

    best_valid_loss = 1e+6

    for i in range(epochs):

        print(f"\nEpoch: {i+1}")

        train_loss, train_accuracy, train_labels_score = train_iter(model, train_loader, optimizer, criterion, mode)
        train_losses.append(float(train_loss))
        train_accuracy_scores.append(float(train_accuracy))
        train_labels_scores.append(float(train_labels_score))
        print(f"Train loss: {train_loss}, Train accuracy: {train_accuracy}, Train labels score: {train_labels_score}")

        valid_loss, valid_accuracy, valid_labels_score = evaluate_iter(model, valid_loader, criterion)
        valid_losses.append(float(valid_loss))
        valid_accuracy_scores.append(float(valid_accuracy))
        valid_labels_scores.append(float(valid_labels_score))
        print(f"Valid loss: {valid_loss}, Valid accuracy: {valid_accuracy}, Valid labels score: {valid_labels_score}")

        scheduler.step()
        
        if valid_loss < best_test_loss:
            best_test_loss = valid_loss
            torch.save(model.state_dict(), f"models/TokenCLF/{mode}/state_dict_model.pth")
            torch.save(optimizer.state_dict(), f"models/TokenCLF/{mode}/state_dict_optimizer.pth")
            torch.save(scheduler.state_dict(), f"models/TokenCLF/{mode}/state_dict_scheduler.pth")
            print("Saved to:", i+1, "\n")
        
        with open(f'drive/MyDrive/projects/KONTUR/models/TokenCLF/{mode}/info.json', 'w') as file_object:
                    info = {
                        'train_losses': train_losses,
                        'train_accuracy': train_accuracy_scores,
                        'train_labels_evals': train_labels_scores,
                        'valid_losses': valid_losses,
                        'valid_accuracy': valid_accuracy_scores,
                        'valid_start_evals': valid_labels_scores
                    }
                    file_object.write(json.dumps(info, indent=2))
