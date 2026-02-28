import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'xxx'
import math
import argparse
import torch
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import pandas as pd

from dataset import MyDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate,read_split_data

class SparseOptimizer(optim.SGD):
    def __init__(self, params, lr=1e-3,  momentum=0.9,weight_decay=5E-5, zero_mask=None, index_map=None):
        super(SparseOptimizer, self).__init__(params, lr=lr, momentum=momentum,weight_decay=weight_decay)
        self.zero_mask = zero_mask
        self.index_map = index_map
        self.total_update_count = 0
        self.last_step_update = 0
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        step_updates = 0

        for group in self.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                mask = self.zero_mask[idx].to(p.device)
                update_mask = mask & (p.grad != 0)
                step_updates += update_mask.sum().item()
                p.data[mask] -= group['lr'] * p.grad[mask]

        self.last_step_update = step_updates
        self.total_update_count += step_updates

        return loss

def generate_zero_mask_and_index(weights_dict):
    zero_mask = []
    index_map = {}
    index = 0
    for name, param in weights_dict.items():
        if "head" in name:
            zero_mask.append(torch.ones_like(param, dtype=torch.bool))
        else:
            zero_mask.append(param == 0)
        index_map[index] = name
        index += 1

    return zero_mask, index_map

def save_summary(epoch, metrics, filename='1_summary.csv'):
    df = pd.DataFrame([metrics])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

def count_parameters(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()
    return total_params, zero_params
def main(args):
    spweight='xxx_best.pth'
    spweight = torch.load(spweight)['state_dict']
    zero_mask, index_map = generate_zero_mask_and_index(spweight)
    total_w = 0
    zero_w = 0
    for name, p in spweight.items():
        num = p.numel()
        z = (p == 0).sum().item()
        total_w += num
        zero_w += z
        print(f"[SP WEIGHT] {name:40s}  zeros={z}  nonzeros={num - z}")

    print(f"\n>>> Original Sparse Weight File:")
    print(f"    Total params   : {total_w}")
    print(f"    Zero params    : {zero_w}")
    print(f"    Sparse ratio   : {zero_w / total_w * 100:.2f}%\n")
    mask_total = 0
    mask_true = 0
    for m in zero_mask:
        mask_total += m.numel()
        mask_true += m.sum().item()

    print(f">>> zero_mask summary:")
    print(f"    Total positions     : {mask_total}")
    print(f"    Updatable positions : {mask_true}")
    print(f"    Frozen positions    : {mask_total - mask_true}")
    print(f"    Trainable ratio     : {mask_true / mask_total * 100:.2f}%\n")

    minloss = 9999
    MAXmAUC = 0
    MAXmAP = 0
    MAXmf1 = 0
    MAXmprec = 0
    MAXmrecall = 0
    MAXmacc = 0
    minepoch = 0
    device=args.device
    print(torch.cuda.device_count())
    if os.path.exists("./weights_sp") is False:
        os.makedirs("./weights_sp")

    tb_writer = SummaryWriter(f"./logs/sp/")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path1,
                                                                                                args.data_path2)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}


    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])


    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)


    model = create_model(num_classes=args.num_classes,has_logits=False)
    model = model.to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        ckpt = torch.load(args.weights, map_location="cpu")

        if isinstance(ckpt, dict) and "model" in ckpt:
            print(">>> Using checkpoint['model']")
            weights = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            print(">>> Using checkpoint['state_dict']")
            weights = ckpt["state_dict"]
        else:
            print(">>> Using checkpoint directly as state_dict")
            weights = ckpt

        del_keys = ["head.weight", "head.bias"]
        for k in del_keys:
            if k in weights:
                print(f"Deleting key from pretrained weights: {k}")
                del weights[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)

        print("\n====== Weight Load Verification (ViT) ======")
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        print(f"Successfully loaded parameters: {len(weights.keys())}")
        print("======================================\n")

    total_params, zero_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Zero parameters: {zero_params}")

    if args.freeze_layers:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad_(False)
            else:
                print("Training {}".format(name))

    optimizer = SparseOptimizer(model.parameters(), lr=args.lr, zero_mask=zero_mask, index_map=index_map)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()
        val_loss, val_acc, mAUC, mf1, mprec, mrecall, map,metrics_dict = evaluate(model=model,
                                                                     data_loader=val_loader,
                                                                     device=device,
                                                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "mAUC", "mf1", "mprec", "mrecall",
                "map"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], mAUC, epoch)
        tb_writer.add_scalar(tags[6], mf1, epoch)
        tb_writer.add_scalar(tags[7], mprec, epoch)
        tb_writer.add_scalar(tags[8], mrecall, epoch)
        tb_writer.add_scalar(tags[9], map, epoch)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mAUC": mAUC,
            "mf1": mf1,
            "mprec": mprec,
            "mrecall": mrecall,
            "map": map
        }
        save_summary(epoch, metrics,filename='xxx_summary.csv')
        print("mAUC:{:.3f}".format(mAUC))
        print("mAP:{:.3f}".format(map))
        print("mf1:{:.3f}".format(mf1))
        if val_loss < minloss:
            print(epoch)
            minepoch = epoch
            minloss = val_loss
            torch.save(model.state_dict(), f"xxx.pth")_
        print(minepoch)
        if MAXmacc < val_acc:
            MAXmacc = val_acc
        if MAXmAUC < mAUC:
            MAXmAUC = mAUC
        if MAXmf1 < mf1:
            MAXmf1 = mf1
        if MAXmAP < map:
            MAXmAP = map
        if MAXmprec < mprec:
            MAXmprec = mprec
        if MAXmrecall < mrecall:
            MAXmrecall = mrecall

        data = {
            'dataset': ['xxx'],
            'MAXmAUC': [MAXmAUC],
            'MAXmAP': [MAXmAP],
            'MAXmf1': [MAXmf1],
            'MAXmprec': [MAXmprec],
            'MAXmrecall': [MAXmrecall],
            'MAXmacc': [MAXmacc]
        }

        if os.path.exists(f'xxx.csv'):
            os.remove(f'xxx.csv')

        df = pd.DataFrame(data)
        df.to_csv(f'xxx.csv', mode='a', header=False, index=False)
    print(f"\n>> Total updated parameters across all epochs: {optimizer.total_update_count}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.001)
    parser.add_argument('--data-path1', type=str, default="")
    parser.add_argument('--data-path2', type=str,default="")
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--weights', type=str, default='',help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
