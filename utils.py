import os
import json
import random
import torch
import sys
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def calculate_euclidean_distance(x):

    x_norm2 = (x ** 2).sum(dim=-1, keepdim=True)

    dot = torch.matmul(x, x.transpose(1, 2))
    dist2 = x_norm2 + x_norm2.transpose(1, 2) - 2 * dot
    dist2 = torch.clamp(dist2, min=0.0)

    return dist2

def safe_norm_heat_kernel(dist2, t=1.0, zero_diag=True, eps=1e-12):
    sim = torch.exp(-dist2 / t)
    if zero_diag:
        diag = torch.diagonal(sim, dim1=-2, dim2=-1)
        sim = sim - torch.diag_embed(diag)

    return sim

def read_split_data(root1: str,root2: str):
    random.seed(0)
    assert os.path.exists(root1), "dataset root: {} does not exist.".format(root1)
    assert os.path.exists(root2), "dataset root: {} does not exist.".format(root2)
    train_class = [cla for cla in os.listdir(root1) if os.path.isdir(os.path.join(root1, cla))]
    val_class = [cla for cla in os.listdir(root1) if os.path.isdir(os.path.join(root1, cla))]
    train_class.sort()
    val_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(train_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num1 = []
    every_class_num2 = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in train_class:
        cla_path = os.path.join(root1, cla)
        trainimages = [os.path.join(root1, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        trainimages.sort()
        image_class = class_indices[cla]
        every_class_num1.append(len(trainimages))
        for img_path in trainimages:
            train_images_path.append(img_path)
            train_images_label.append(image_class)
    for cla in val_class:
        cla_path = os.path.join(root2, cla)
        valimages = [os.path.join(root2, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        valimages.sort()
        image_class = class_indices[cla]
        every_class_num2.append(len(valimages))
        for img_path in valimages:
            val_images_path.append(img_path)
            val_images_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_class_num1)+sum(every_class_num2)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):

        images, labels = data
        sample_num += images.shape[0]
        images = images.to(device)
        labels = labels.to(device)
        pred,patch_embedding,output = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        output = output[:, 1:, :]
        B, C, H, W = images.shape
        raw_x = F.adaptive_avg_pool2d(images, (14, 14))
        raw_x = raw_x.view(B, C, -1).permute(0, 2, 1)
        pre_dist = calculate_euclidean_distance(raw_x)
        post_dist = calculate_euclidean_distance(output)
        pre_W = safe_norm_heat_kernel(pre_dist, t=1.0)
        post_W = safe_norm_heat_kernel(post_dist, t=1.0)

        similarity_loss = F.mse_loss(pre_W, post_W)

        loss = loss_function(pred, labels.to(device))+similarity_loss*0.01

        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

import numpy as np
from sklearn.metrics import classification_report, average_precision_score,roc_auc_score,precision_recall_fscore_support,confusion_matrix,accuracy_score
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    all_labels = []
    all_preds = []
    all_scores = []
    all_scores2 = []
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred,_,_ = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_classes.cpu().numpy())
        all_scores.extend(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())
        all_scores2.extend(torch.nn.functional.softmax(pred, dim=1)[:,1].cpu().numpy())
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    print(np.array(all_scores).shape)
    if np.array(all_scores).shape[1]==2:
        auc = roc_auc_score(all_labels, all_scores2, average=None, multi_class='ovr')
        ap_score = average_precision_score(all_labels, all_scores2, average=None)
    else:
        auc = roc_auc_score(all_labels, all_scores, average=None, multi_class='ovr')
        ap_score = average_precision_score(all_labels, all_scores, average=None)

    prec,recall,f1,_=precision_recall_fscore_support(all_labels, all_preds,zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    num_classes = cm.shape[0]
    tp = cm.diagonal()
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    for i in range(num_classes):
        print(f"Class {i}: TP={tp[i]}, TN={tn[i]}, FP={fp[i]}, FN={fn[i]}")
    metrics_dict = {
        "tp": tp.tolist(),
        "tn": tn.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist()
    }
    total_tp = tp.sum()
    total_tn = tn.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    print(f"Total TP={total_tp}, TN={total_tn}, FP={total_fp}, FN={total_fn}")
    if num_classes == 2:
        mAUC_macro = roc_auc_score(all_labels, all_scores2)
        mAUC_micro = roc_auc_score(all_labels, all_scores2, average="micro")
        mAP_macro = average_precision_score(all_labels, all_scores2)
        mAP_micro = average_precision_score(all_labels, all_scores2, average="micro")
        prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )
        prec_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="micro", zero_division=0
        )
    else:
        mAUC_macro = roc_auc_score(all_labels, all_scores, average="macro", multi_class='ovr')
        mAUC_micro = roc_auc_score(all_labels, all_scores, average="micro", multi_class='ovr')
        mAP_macro = average_precision_score(all_labels, all_scores, average="macro")
        mAP_micro = average_precision_score(all_labels, all_scores, average="micro")
        prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        prec_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="micro", zero_division=0
        )

    acc = accuracy_score(all_labels, all_preds)
    print(
        f"[Epoch {epoch}] "
        f"Loss: {accu_loss.item() / (step + 1):.4f} | "
        f"Acc: {acc:.4f} | "
        f"mAUC (macro/micro): {mAUC_macro:.4f}/{mAUC_micro:.4f} | "
        f"mAP (macro/micro): {mAP_macro:.4f}/{mAP_micro:.4f} | "
        f"F1 (macro/micro): {f1_macro:.4f}/{f1_micro:.4f} | "
        f"Prec (macro/micro): {prec_macro:.4f}/{prec_micro:.4f} | "
        f"Recall (macro/micro): {recall_macro:.4f}/{recall_micro:.4f}"
    )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,auc.mean(),f1.mean(),prec.mean(),recall.mean(),ap_score.mean(),metrics_dict
