
import torch
from tqdm import tqdm
#from configs import classnames, index2name

def run_epoch(phase, model, criterion, optimizer, scheduler, dataloader, device):

        class_to_idx = dataloader.dataset.class_to_idx
        idx_to_class = {val: key for key, val in class_to_idx.items()}

        print(phase.upper())
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0.0
        running_wrongs = 0.0

        running_class_stats = {classname: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'num_preds': 0, 'num_gt': 0} for classname in
                               dataloader.dataset.classes}

        running_class_corrects = {i: 0 for i in range(5)}
        running_class_wrongs = {i: 0 for i in range(5)}

        # Iterate over data once.
        batchidx = 0
        for inputs, labels in tqdm(dataloader):
            # for i_step in tqdm(range(steps_per_epoch), desc='step'):
            #    inputs, labels = next(dataloaders[phase])

            # batchidx += 1
            # if batchidx > 100:
            #     print(" ")
            #     break

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    #before = list(model.parameters())[-2].sum()
                    optimizer.step()
                    #after = list(model.parameters())[-2].sum()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_wrongs += torch.sum(preds != labels.data)
            # extended statistics
            for gt in range(5):
                classname = idx_to_class[gt]
                running_class_stats[classname]['TP'] += int(torch.sum((preds == gt) & (labels == gt)))
                running_class_stats[classname]['TN'] += int(torch.sum((preds != gt) & (labels != gt)))
                running_class_stats[classname]['FP'] += int(torch.sum((preds == gt) & (labels != gt)))
                running_class_stats[classname]['FN'] += int(torch.sum((preds != gt) & (labels == gt)))
                running_class_stats[classname]['num_preds'] += int(torch.sum((preds == gt)))
                running_class_stats[classname]['num_gt'] += int(torch.sum((labels == gt)))

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = float(running_corrects) / (float(running_corrects) + float(running_wrongs) + 1)

        class_acc = {i: float(running_class_corrects[i]) / (
                    float(running_class_corrects[i]) + float(running_class_wrongs[i]) + 1e-6) for i in range(5)}

        return running_class_stats, epoch_loss, epoch_acc, class_acc