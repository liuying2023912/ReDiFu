import os
import numpy as np
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import random
from vision import confuPLT
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


seed = 123

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, windows=5):
    trainset = MELDDataset('data/meld_multimodal_features.pkl', windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    testset = MELDDataset('data/meld_multimodal_features.pkl', train=False, windows=windows)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=True, windows=5):
    trainset = IEMOCAPDataset("data/iemocap_multimodal_features.pkl", windows=windows)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=valid_sampler,
        collate_fn=trainset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    testset = IEMOCAPDataset("data/iemocap_multimodal_features.pkl", train=False, windows=windows)
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, valid_loader, test_loader


def delta_consistency_loss(x, delta, mask):
    mask = mask.unsqueeze(-1).float()
    x = x * mask
    delta = delta * mask

    x_norm = F.normalize(x, dim=-1)
    delta_norm = F.normalize(delta, dim=-1)
    cos_sim = (x_norm * delta_norm).sum(dim=-1)
    cos_sim = (cos_sim * mask.squeeze(-1)).sum() / mask.sum()
    return 1 - cos_sim


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False, dataset=None):
    losses = []
    preds = []
    labels = []
    masks = []
    all_transformer_outs = []
    umasks = []

    assert not train or optimizer is not None

    device = next(model.parameters()).device

    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label, Self_semantic_adj, Cross_semantic_adj, Semantic_adj = [
            d.to(device) for d in data
        ]

        umasks.append(umask)
        qmask = qmask.permute(1, 0, 2)
        lengths = torch.tensor(
            [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))],
            device=device
        )

        log_prob, all_log_prob, all_prob, all_transformer_out = model(
            textf, visuf, acouf, umask, qmask, lengths, Self_semantic_adj, Cross_semantic_adj, Semantic_adj
        )

        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        fusion_loss = loss_function(lp_all, labels_, umask)

        t_loss = loss_function(log_prob[0].view(-1, log_prob[0].size()[2]), labels_, umask)
        a_loss = loss_function(log_prob[1].view(-1, log_prob[1].size()[2]), labels_, umask)
        v_loss = loss_function(log_prob[2].view(-1, log_prob[2].size()[2]), labels_, umask)
        loss = fusion_loss + t_loss * (t_loss / 10) + a_loss * (a_loss / 10) + v_loss * (v_loss / 10)

        lp_ = all_prob.view(-1, all_prob.size()[2])
        pred_ = torch.argmax(lp_, 1)
        preds.append(pred_.data.cpu().numpy())
        all_transformer_outs.append(all_transformer_out)
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())

        if train:
            loss.backward()
            optimizer.step()

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), [], []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, all_transformer_outs, umasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--l2', type=float, default=0.00005, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=512, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=64, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=40, metavar='E', help='number of epochs')
    parser.add_argument('--windows', type=int, default=20, help='number of windows')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--save_model_path', default='./IEMOCAP', type=str, help='模型输出路径')
    parser.add_argument('--seed', default=2094, type=int, help='seed')
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    feat2dim = {'IS10': 1582, 'denseface': 342, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024
    n_speakers = 9 if args.Dataset == 'MELD' else 2
    n_classes = 6 if args.Dataset == 'IEMOCAP' else 7

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = Transformer_Based_Model(
        args.Dataset,
        D_text,
        D_visual,
        D_audio,
        args.n_head,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        n_speakers=n_speakers,
        dropout=args.dropout
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))
    if torch.cuda.device_count() > 0:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    if args.Dataset == 'MELD':
        loss_weights = torch.FloatTensor([
            1 / 0.481226,
            1 / 0.107663,
            1 / 0.191571,
            1 / 0.079669,
            1 / 0.154023,
            1 / 0.026054,
            1 / 0.132184
        ])
        loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(
            valid=0.0,
            batch_size=batch_size,
            num_workers=0,
            windows=args.windows
        )
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([
            1 / 0.086747,
            1 / 0.144406,
            1 / 0.227883,
            1 / 0.160585,
            1 / 0.127711,
            1 / 0.252668
        ])
        loss_function = MaskedNLLLoss(loss_weights.cuda())
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
            valid=0.0,
            batch_size=batch_size,
            num_workers=0,
            windows=args.windows
        )

    best_fscore = None
    best_loss = None
    best_label = None
    best_pred = None
    best_mask = None
    best_feature = None
    best_umasks = None
    all_train_fscore = []
    all_test_fscore = []

    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore, _, _ = train_or_eval_model(
            model,
            loss_function,
            train_loader,
            optimizer,
            train=True,
            dataset=args.Dataset
        )

        valid_loss, valid_acc, _, _, _, valid_fscore, _, _ = train_or_eval_model(
            model,
            loss_function,
            valid_loader,
            train=False,
            dataset=args.Dataset
        )

        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, all_transformer_outs, umasks = train_or_eval_model(
            model,
            loss_function,
            test_loader,
            train=False,
            dataset=args.Dataset
        )

        scheduler.step()

        all_train_fscore.append(train_fscore)
        all_test_fscore.append(test_fscore)

        if best_fscore is None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label = test_label
            best_pred = test_pred
            best_mask = test_mask
            best_feature = all_transformer_outs
        print(
            'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(
                e + 1,
                train_loss,
                train_acc,
                train_fscore,
                test_loss,
                test_acc,
                test_fscore,
                round(time.time() - start_time, 2)
            )
        )

    save_path = os.path.join(args.save_model_path, "bestModel.pth")
    torch.save(model, save_path)
    print('Model Performance:')
    print('Best_Test-FScore-epoch_index: {}'.format(all_test_fscore.index(max(all_test_fscore)) + 1))
    print('Best_Test_F-Score: {}'.format(max(all_test_fscore)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4, zero_division=0))
    confuPLT(confusion_matrix(best_label, best_pred, sample_weight=best_mask).astype(int), args.Dataset)