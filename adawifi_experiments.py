import os
import copy
import scipy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from adawifi_data_utils import *
from model import *

logging.basicConfig(level=logging.INFO)

device = torch.device('cpu')
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    device = torch.device(num_devices - 1)
print(f'using device: {device}')

all_rooms = ['Floor11', 'Room2', 'MeetingRoom']
target_gestures = ['Swipe', 'Push-Pull', 'Clap', 'Slide', 'Draw-Z', 'Draw-O']
room_users = {
    'Floor11': ['yuanchun', 'naiyu', 'lixiangyu', 'zhuxiangyu', 'kongrui'],
    # 'Office': ['yuanchun', 'yuanzhe', 'naiyu', 'yongliang', 'hanfei'],
    'Room2': ['yuanzhe', 'yuanchun', 'naiyu', 'wenhao', 'wangxiang'],
    'MeetingRoom': ['yuanchun', 'yuanzhe', 'naiyu', 'sunyi', 'yanghuan']
}
data_root = '/data3/adawifi/ants_data'


def load_data(rooms, users=None, split_head=True, split_ratio=0.8, skip=1):
    """
    Load data from specific rooms and users
    If split_head is True, return the head portion of samples.
    If split_head is False, return the tail portion of samples.
    """
    if not rooms:
        rooms = [0,1,2]
    if not users:
        users = [0,1,2,3,4]
    all_data_df = []
    for room in rooms:
        for user in users:
            room_name = all_rooms[room]
            user_name = room_users[room_name][user]
            data_df = DfsData(f'{data_root}/{room_name}/{user_name}/dfs').df
            data_df['user'] = user
            data_df['user_name'] = user_name
            data_df['room'] = room
            data_df['room_name'] = room_name
            # print(data_df)
            data_df.sort_values(by='timestamp', inplace=True, ascending=True)
            data_df = data_df.reset_index().loc[::skip]
            if split_head:
                data_df_split = data_df[0:int(len(data_df) * split_ratio)]
            else:
                data_df_split = data_df[int(len(data_df) * split_ratio):]
            all_data_df.append(data_df_split)
    all_data_df = pd.concat(all_data_df)
    all_data_ds = MyDfsDataset(df=all_data_df, target_gestures=target_gestures, cache_dir='/Users/naiyu/Desktop/Data/data/my_dfs_cache')
    return all_data_df, all_data_ds


def get_n_shots_ds(old_ds, n=1):
    """
    Get the n-shot training dataset,
    i.e. select the first n samples for each gesture from the original dataset (old_ds)
    """
    old_df = old_ds.df
    indices = np.arange(n)
    new_df = old_df.sort_values(by='timestamp', ascending=True).groupby('gesture').nth(indices).reset_index()
    import copy
    new_ds = copy.copy(old_ds)
    new_ds.df = new_df
    new_ds.all_samples = new_ds.load_data(new_df)
    return new_ds


def train_model(
    model, train_ds, valid_ds, test_ds, output_dir='temp', tag='none', device=None,
    n_epochs=50, lr=0.001, weight_decay=0.01, batch_size=32, feature='csi', target='activity_id', verbose=True):
    """
    @param feature could be csi, inp, or bvp
    @param target could be gesture_id, room, location, orientation, etc.
    """
    if device is None:
        device = torch.device('cpu')
    
    if verbose:
        print('=' * 80)
    print(f'train_encoder: {tag}, model: {model.__class__.__name__}, device: {device}')
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    cls_loss_fn = torch.nn.CrossEntropyLoss()
        
    os.makedirs(os.path.join(output_dir, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(output_dir, f'runs/{tag}_{timestamp}'))

    dataiter = iter(test_dl)
    sample = dataiter.next()
        
    if writer:
        model.eval()
        inputs = sample[feature].to(device)
        writer.add_graph(model, inputs)
        writer.flush()

    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        for i, sample in tqdm(enumerate(train_dl), desc=f'Epoch {epoch_index}', total=len(train_dl), disable=(not verbose)):
            inputs, labels = sample[feature].to(device), sample[target].to(device)
            optimizer.zero_grad()
            outputs, enc = model(inputs)
            agg_enc = enc.mean(dim=1)
            cls_loss = cls_loss_fn(outputs, labels)
            loss = cls_loss
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        return running_loss / (i + 1)

    def evaluate(eval_dl, n_samples=None):
        running_vloss = 0.0
        n_correct = 0
        n_total = 0
        for i, sample in enumerate(eval_dl):
            inputs, labels = sample[feature].to(device), sample[target].to(device)
            outputs, enc = model(inputs)
            
            agg_enc = F.normalize(enc.mean(dim=1), dim=1)
            cls_loss = cls_loss_fn(outputs, labels)
            loss = cls_loss
            
            running_vloss += loss.item()
            preds = torch.argmax(outputs, axis=-1)
            n_correct += torch.sum(labels == preds)
            n_total += labels.size(0)
            if n_samples and n_total >= n_samples:
                break

        avg_loss = running_vloss / (i + 1)
        acc = n_correct * 100 / n_total
        return avg_loss, acc.item(), n_total

    best_vloss = 1_000_000.
    train_ACCs = []
    valid_ACCs = []
    valid_LOSSs = []
    test_ACCs = []
    for epoch in tqdm(range(1, n_epochs + 1), desc='Training progress', disable=verbose):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_one_epoch(epoch, writer)
        model.train(False)

        train_loss, train_acc, train_total = evaluate(train_dl, 600)
        valid_loss, valid_acc, valid_total = evaluate(valid_dl, 600)
        test_loss, test_acc, test_total = evaluate(test_dl)
        if verbose:
            print(f'TRAIN loss {train_loss:.4f} acc {(train_acc):.2f} ({train_total} samples)')
            print(f'VALID loss {valid_loss:.4f} acc {(valid_acc):.2f} ({valid_total} samples)')
            print(f'TEST  loss {test_loss:.4f} acc {(test_acc):.2f} ({test_total} samples)')
        train_ACCs.append(train_acc)
        valid_LOSSs.append(valid_loss)
        valid_ACCs.append(valid_acc)
        test_ACCs.append(test_acc)

        # Log the running loss averaged per batch
        if writer:
            writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss, 'test': test_loss}, epoch)
            writer.add_scalars('acc', {
                'train': train_acc, 'valid': valid_acc, 'test': test_acc, 'test_at_min_loss': test_ACCs[np.argmin(valid_LOSSs)],
                'max_train': max(train_ACCs), 'max_valid': max(valid_ACCs), 'max_test': max(test_ACCs),
            }, epoch)
            writer.flush()

        # Track best performance, and save the model's state
        if valid_loss < best_vloss:
            best_vloss = valid_loss
            model_path = f'temp/models/{tag}_{timestamp}'
            torch.save(model.state_dict(), model_path)
            
    test_at_min_loss = test_ACCs[np.argmin(valid_LOSSs)]
    if verbose:
        print('-' * 80)
        print(f'Finished experiment {tag}')
    print(f'Accuracy: test_at_min_loss={test_at_min_loss:.2f}, max_test={max(test_ACCs):.2f}, ' +
          f'max_train={max(train_ACCs):.2f}, max_valid={max(valid_ACCs):.2f}')
    model.load_state_dict(torch.load(f'temp/models/{tag}_{timestamp}'))
    return model, test_at_min_loss, max(test_ACCs)


def train_model_wrapper(model_cls, n_activities=6, n_feature=60, tag=None, **kwargs):
    tag = tag if tag else model_cls.__name__
    model = model_cls(n_activities=n_activities, n_feature=n_feature)
    return train_model(
        model,
        device=device, tag=tag, **kwargs
    )


class Assign_Rx_Pos(object):
    """
    This class is a transformation function that assigns a position id to each client in each environment.
    The position id is used by our model during adaptation.
    """
    def __init__(self, datasets, granularity='room'):
        self.datasets = datasets
        self.granularity = granularity
        self.pos2idx = {}
        self.max_pos_idx = 0
        self._build_pos2idx_mapping()

    def _get_domain_id(self, sample):
        if self.granularity == 'room':
            return sample['room']
        else:
            return sample['room'] * 10 + sample['user']

    def _build_pos2idx_mapping(self):
        pos_idx = 1
        for ds in self.datasets:
            for sample in ds.all_samples:
                domain_id = self._get_domain_id(sample)
                n_rx = sample['dfs'].shape[0]
                for i in range(n_rx):
                    pos = int(domain_id * 10 + i)
                    if pos not in self.pos2idx:
                        self.pos2idx[pos] = pos_idx
                        pos_idx += 1
        self.max_pos_idx = pos_idx

    def _add_pos_ids(self, sample, transformed_sample):
        domain_id = self._get_domain_id(sample)
        n_rx = sample['dfs'].shape[0]
        rx_pos_ids = []
        for i in range(n_rx):
            rx_pos = domain_id * 10 + i
            rx_pos_ids.append(self.pos2idx[rx_pos])
        rx_pos_ids = np.array(rx_pos_ids)
        transformed_sample['pos'] = torch.from_numpy(rx_pos_ids).long()
        return transformed_sample
    
    def augment_transform(self, sample):
        transformed_sample = MyDfsDataset.augment_transform(sample)
        return self._add_pos_ids(sample, transformed_sample)
    
    def default_transform(self, sample):
        transformed_sample = MyDfsDataset.default_transform(sample)
        return self._add_pos_ids(sample, transformed_sample)


def adapt_MixRx_Transformer(
        model, source_model, train_ds=None, valid_ds=None, test_ds=None, tune_ds=None,
        output_dir='temp', tag='T', device=None, loss_mode='source+target+mix+inter', use_pos=True,
        n_epochs=50, lr=0.001, weight_decay=0, batch_size=32, feature='dfs', target='gesture_id', verbose=True):
    """
    @param feature could be bvp, dfs, or inp
    @param target could be gesture_id, room, location, orientation, etc.
    """
    print(f'adapt_model tag: {tag}, model: {model.__class__.__name__}, device: {device}')
    if device is None:
        device = torch.device('cpu')
    if verbose:
        print('=' * 80)
    os.makedirs(os.path.join(output_dir, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(output_dir, f'runs/{tag}_{timestamp}'))
    cls_loss_fn = torch.nn.CrossEntropyLoss()

    def inter_dist_loss_fn(emb_agg, labels):
        batch_sz = labels.size(0)
        emb_dist_mat = emb_agg.matmul(emb_agg.transpose(0, 1))
        labels = labels.view(batch_sz, 1).float()
        label_dist_mat = (torch.cdist(labels, labels, p=2.0) == 0) * 1.0
        loss = F.binary_cross_entropy_with_logits(emb_dist_mat, label_dist_mat)
        return loss

    def cross_dist_loss_fn(emb_agg, emb_agg_, labels, labels_):
        emb_dist_mat = emb_agg.matmul(emb_agg_.transpose(0, 1))
        labels = labels.view(labels.size(0), 1).float()
        labels_ = labels_.view(labels_.size(0), 1).float()
        label_dist_mat = (torch.cdist(labels, labels_, p=2.0) == 0) * 1.0
        loss = F.binary_cross_entropy_with_logits(emb_dist_mat, label_dist_mat)
        # print(loss, enc_dist_mat.shape, label_dist_mat.shape)
        return loss

    def loss_fn(
            emb, emb_agg, outputs, labels,
            emb_, emb_agg_, outputs_, labels_,
            emb_mix, emb_agg_mix, outputs_mix, labels_mix,
            loss_mode=loss_mode, print_loss=False):

        cls_loss = cls_loss_fn(outputs, labels)
        cls_loss_ = cls_loss_fn(outputs_, labels_)
        cls_loss_mix = cls_loss_fn(outputs_mix, labels_mix)

        all_emb_agg = torch.cat((emb_agg, emb_agg_, emb_agg_mix), dim=0)
        all_labels = torch.cat((labels, labels_, labels_mix), dim=0)
        inter_dist = inter_dist_loss_fn(all_emb_agg, all_labels)
        # inter_dist = cross_dist_loss_fn(emb_agg, emb_agg_, labels, labels_)

        loss = 0
        if 'source' in loss_mode and (not torch.isnan(cls_loss)):
            loss += 1.0 * cls_loss
        if 'target' in loss_mode and (not torch.isnan(cls_loss_)):
            loss += 1.0 * cls_loss_
        if 'mix' in loss_mode and (not torch.isnan(cls_loss_mix)):
            loss += 1.0 * cls_loss_mix
        if 'inter' in loss_mode and (not torch.isnan(inter_dist)):
            loss += 1.0 * inter_dist

        if print_loss:
            print(outputs_mix.shape)
            print(
                f'loss={loss:4f}, cls_loss={cls_loss:4f}, cls_loss_={cls_loss_:4f}, cls_loss_mix={cls_loss_mix:4f}, inter_dist={inter_dist:4f}')
        return loss

    def loss_fn_train(emb, emb_agg, outputs, labels, loss_mode=loss_mode, print_loss=False):
        cls_loss = cls_loss_fn(outputs, labels)
        inter_dist = inter_dist_loss_fn(emb_agg, labels)
        loss = 0
        if 'source' in loss_mode:
            loss += cls_loss
        if 'inter' in loss_mode:
            loss += inter_dist
        if print_loss:
            print(f'loss={loss:4f}, cls_loss={cls_loss:4f}, inter_dist={inter_dist:4f}')
        return loss

    # source_model = source_model.to(device)
    # source_model.eval()
    model = model.to(device)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=MyDfsDataset.default_collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=MyDfsDataset.default_collate_fn)
    tune_dl = DataLoader(tune_ds, batch_size=batch_size, shuffle=True, collate_fn=MyDfsDataset.default_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=MyDfsDataset.default_collate_fn)

    dataiter = iter(test_dl)
    # sample = dataiter.next()
    sample = dataiter.__next__()
    n_test_rx = sample[feature].size(1)
    if writer:
        model.eval()
        inputs = sample[feature].to(device)
        writer.add_graph(model, inputs)
        writer.flush()

    def run_one_epoch(dl, loss_mode=loss_mode, epoch=0, evaluate=False, n_samples=None, tb_writer=None, print_loss=False):
        running_loss = 0.
        n_correct = 0
        n_total = 0
        if not evaluate:
            if source_model:
                if epoch <= n_epochs * 0.3333:
                    optimizer = torch.optim.Adam([
                        {'params': model.pos_weight},
                    ], lr=lr * 10, weight_decay=weight_decay)
                else:
                    optimizer = torch.optim.Adam([
                        {'params': model.client_encoder.parameters()},
                        {'params': model.transformer_encoder.parameters()},
                        {'params': model.classifier.parameters()},
                        {'params': model.cls_emb},
                        {'params': model.pos_emb},
                        # {'params': model.pos_weight, 'lr': lr*10},
                    ], lr=lr, weight_decay=weight_decay)
            else:
                if epoch <= n_epochs * 0.6666:
                    optimizer = torch.optim.Adam([
                        {'params': model.client_encoder.parameters()},
                        {'params': model.transformer_encoder.parameters()},
                        {'params': model.classifier.parameters()},
                        {'params': model.cls_emb},
                        # {'params': model.pos_emb},
                        # {'params': model.pos_weight, 'lr': lr*10},
                    ], lr=lr, weight_decay=weight_decay)
                else:
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for idx, sample in tqdm(enumerate(dl), desc=f'Epoch {epoch}', total=len(dl), disable=(evaluate or (not verbose))):
            inputs, labels, data_T, pos = sample[feature].to(device), sample[target].to(device), sample['data_T'], sample['pos']
            current_batch_sz = inputs.size(0)
            sample_ = iter(tune_dl).__next__()
            inputs_, labels_, data_T_, pos_ = sample_[feature].to(device), sample_[target].to(device), sample_[
                'data_T'], sample_['pos']

            if not use_pos:
                pos = None
                pos_ = None

            if not evaluate:
                optimizer.zero_grad()

            if source_model:
                # adaption mode
                emb = model.forward_client_emb(inputs, data_T, pos)
                emb_ = model.forward_client_emb(inputs_, data_T_, pos_)
                # onehot_labels = F.one_hot(labels, num_classes=model.n_activities)
                emb_mix = []
                labels_mix = []
                for i in range(labels_.size(0)):
                    i_label_ = labels_[i]
                    same_label_indices = (labels == i_label_).nonzero()[:, 0]
                    same_label_emb = emb[same_label_indices]
                    # print(same_label_emb.shape)
                    repeat_i_emb_ = emb_[i:i + 1].repeat((same_label_emb.size(0), 1, 1))
                    # print(same_label_indices.shape, same_label_emb.shape, repeat_i_emb_.shape)
                    emb_mix_i = torch.cat((same_label_emb, repeat_i_emb_), dim=1)
                    labels_mix_i = labels[same_label_indices]
                    emb_mix.append(emb_mix_i)
                    labels_mix.append(labels_mix_i)
                # emb_mix is the embeddings mixing both source-domain clients and target-domain clients
                emb_mix = torch.cat(emb_mix, dim=0)
                labels_mix = torch.cat(labels_mix, dim=0)
                if emb_mix.size(0) > current_batch_sz:
                    select_indices = np.random.randint(emb_mix.size(0), size=current_batch_sz)
                    emb_mix = emb_mix[select_indices, :, :]
                    labels_mix = labels_mix[select_indices]
                rx_indices = np.random.randint(emb_mix.size(1), size=n_test_rx)
                emb_mix = emb_mix[:, rx_indices, :]
                # print(emb_mix.shape, labels_mix.shape)

                emb_agg = model.forward_emb_agg(emb)
                emb_agg_ = model.forward_emb_agg(emb_)
                emb_agg_mix = model.forward_emb_agg(emb_mix)
                outputs = model.classifier(emb_agg)
                outputs_ = model.classifier(emb_agg_)
                outputs_mix = model.classifier(emb_agg_mix)
                loss = loss_fn(
                    emb, emb_agg, outputs, labels,
                    emb_, emb_agg_, outputs_, labels_,
                    emb_mix, emb_agg_mix, outputs_mix, labels_mix,
                    loss_mode=loss_mode, print_loss=print_loss
                )
            else:
                # pre-training mode
                emb = model.forward_client_emb(inputs, data_T, pos)
                emb_agg = model.forward_emb_agg(emb)
                outputs = model.classifier(emb_agg)
                loss = loss_fn_train(emb, emb_agg, outputs, labels, print_loss=print_loss)

            if not evaluate:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, axis=-1)
            n_correct += torch.sum(labels == preds)
            n_total += labels.size(0)
            if n_samples and n_total >= n_samples:
                break
        # FIXME
        # print(model.pos_weight)
        avg_loss = running_loss / (idx + 1)
        acc = n_correct * 100 / n_total
        return avg_loss, acc.item(), n_total

    init_loss, init_acc, init_total = run_one_epoch(test_dl, evaluate=True)
    print(f'INITIAL TEST loss {init_loss:.4f} acc {(init_acc):.2f} ({init_total} samples)')

    best_vloss = 1_000_000.
    train_ACCs = []
    valid_ACCs = []
    valid_LOSSs = []
    test_ACCs = []
    pbar = tqdm(range(1, n_epochs + 1), total=n_epochs, desc=tag, disable=verbose)
    for epoch in pbar:
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss, train_acc, train_total = run_one_epoch(train_dl, epoch=epoch, tb_writer=writer)
        model.train(False)

        valid_loss, valid_acc, valid_total = run_one_epoch(valid_dl, evaluate=True, print_loss=False)  # , loss_mode='source+target+inter')  # loss_mode='source+target+inter'
        test_loss, test_acc, test_total = run_one_epoch(test_dl, evaluate=True)
        if verbose:
            print(f'TRAIN loss {train_loss:.4f} acc {(train_acc):.2f} ({train_total} samples)')
            print(f'VALID loss {valid_loss:.4f} acc {(valid_acc):.2f} ({valid_total} samples)')
            print(f'TEST  loss {test_loss:.4f} acc {(test_acc):.2f} ({test_total} samples)')
        train_ACCs.append(train_acc)
        valid_LOSSs.append(valid_loss)
        valid_ACCs.append(valid_acc)
        test_ACCs.append(test_acc)

        # Log the running loss averaged per batch
        if writer:
            writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss, 'test': test_loss}, epoch)
            writer.add_scalars('acc', {
                'train': train_acc, 'valid': valid_acc, 'test': test_acc,
                'test_at_min_loss': test_ACCs[np.argmin(valid_LOSSs)],
                'max_train': max(train_ACCs), 'max_valid': max(valid_ACCs), 'max_test': max(test_ACCs),
            }, epoch)
            writer.flush()

        # Track best performance, and save the model's state
        if valid_loss < best_vloss:
            best_vloss = valid_loss
            model_path = f'temp/models/{tag}_{timestamp}'
            torch.save(model.state_dict(), model_path)
            
        pbar.update(1)
        pbar.set_postfix(acc=f'{train_acc:.2f} {valid_acc:.2f} {test_acc:.2f}', loss=f'{train_loss:.4f} {valid_loss:.4f} {test_loss:.4f}')

    test_at_min_loss = test_ACCs[np.argmin(valid_LOSSs)]
    if verbose:
        print('-' * 80)
        print(f'Finished experiment {tag}')
    print(f'Accuracy: test_at_min_loss={test_at_min_loss:.2f}, max_test={max(test_ACCs):.2f}, ' +
          f'max_train={max(train_ACCs):.2f}, max_valid={max(valid_ACCs):.2f}')
    model.load_state_dict(torch.load(f'temp/models/{tag}_{timestamp}'))
    return model, test_ACCs[-1], max(test_ACCs), init_acc


def adapt_baseline(
        model, source_model=None, train_ds=None, valid_ds=None, test_ds=None, tune_ds=None,
        output_dir='temp', tag='T', device=None, loss_mode='source+target',
        n_epochs=50, lr=0.001, weight_decay=0, batch_size=32, feature='dfs', target='gesture_id', verbose=True):
    """
    @param feature could be bvp, dfs, or inp
    @param target could be gesture_id, room, location, orientation, etc.
    """
    print(f'adapt_baseline tag: {tag}, model: {model.__class__.__name__}, device: {device}')
    if device is None:
        device = torch.device('cpu')
    if verbose:
        print('=' * 80)
    os.makedirs(os.path.join(output_dir, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(output_dir, f'runs/{tag}_{timestamp}'))
    cls_loss_fn = torch.nn.CrossEntropyLoss()

    # source_model = source_model.to(device)
    # source_model.eval()
    model = model.to(device)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=MyDfsDataset.default_collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=MyDfsDataset.default_collate_fn)
    tune_dl = DataLoader(tune_ds, batch_size=batch_size, shuffle=True, collate_fn=MyDfsDataset.default_collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=MyDfsDataset.default_collate_fn)

    dataiter = iter(test_dl)
    sample = dataiter.__next__()
    n_test_rx = sample[feature].size(1)
    if writer:
        model.eval()
        inputs = sample[feature].to(device)
        writer.add_graph(model, inputs)
        writer.flush()

    def run_one_epoch(dl, loss_mode=loss_mode, epoch=0, evaluate=False, n_samples=None, tb_writer=None, print_loss=False):
        running_loss = 0.
        n_correct = 0
        n_total = 0
        if not evaluate:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for idx, sample in tqdm(enumerate(dl), desc=f'Epoch {epoch}', total=len(dl), disable=(evaluate or (not verbose))):
            inputs, labels, data_T = sample[feature].to(device), sample[target].to(device), sample['data_T']
            current_batch_sz = inputs.size(0)
            sample_ = iter(tune_dl).__next__()
            inputs_, labels_, data_T_ = sample_[feature].to(device), sample_[target].to(device), sample_['data_T']

            if not evaluate:
                optimizer.zero_grad()

            loss = 0
            if 'source' in loss_mode:
                outputs = model(inputs, data_T)
                loss += cls_loss_fn(outputs, labels)
            if 'target' in loss_mode:
                outputs_ = model(inputs_, data_T_)
                loss += cls_loss_fn(outputs_, labels_)

            if not evaluate:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, axis=-1)
            n_correct += torch.sum(labels == preds)
            n_total += labels.size(0)
            if n_samples and n_total >= n_samples:
                break
        avg_loss = running_loss / (idx + 1)
        acc = n_correct * 100 / n_total
        return avg_loss, acc.item(), n_total

    init_loss, init_acc, init_total = run_one_epoch(test_dl, evaluate=True)
    print(f'INITIAL TEST loss {init_loss:.4f} acc {(init_acc):.2f} ({init_total} samples)')

    best_vloss = 1_000_000.
    train_ACCs = []
    valid_ACCs = []
    valid_LOSSs = []
    test_ACCs = []
    
    pbar = tqdm(range(1, n_epochs + 1), total=n_epochs, desc=tag, disable=verbose)
    for epoch in pbar:
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss, train_acc, train_total = run_one_epoch(train_dl, epoch=epoch, tb_writer=writer)
        model.train(False)

        valid_loss, valid_acc, valid_total = run_one_epoch(valid_dl, evaluate=True, print_loss=False)  # , loss_mode='source+target+inter')  # loss_mode='source+target+inter'
        test_loss, test_acc, test_total = run_one_epoch(test_dl, evaluate=True)
        if verbose:
            print(f'TRAIN loss {train_loss:.4f} acc {(train_acc):.2f} ({train_total} samples)')
            print(f'VALID loss {valid_loss:.4f} acc {(valid_acc):.2f} ({valid_total} samples)')
            print(f'TEST  loss {test_loss:.4f} acc {(test_acc):.2f} ({test_total} samples)')
        train_ACCs.append(train_acc)
        valid_LOSSs.append(valid_loss)
        valid_ACCs.append(valid_acc)
        test_ACCs.append(test_acc)

        # Log the running loss averaged per batch
        if writer:
            writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss, 'test': test_loss}, epoch)
            writer.add_scalars('acc', {
                'train': train_acc, 'valid': valid_acc, 'test': test_acc,
                'test_at_min_loss': test_ACCs[np.argmin(valid_LOSSs)],
                'max_train': max(train_ACCs), 'max_valid': max(valid_ACCs), 'max_test': max(test_ACCs),
            }, epoch)
            writer.flush()

        # Track best performance, and save the model's state
        if valid_loss < best_vloss:
            best_vloss = valid_loss
            model_path = f'temp/models/{tag}_{timestamp}'
            torch.save(model.state_dict(), model_path)
            
        pbar.update(1)
        pbar.set_postfix(acc=f'{train_acc:.2f} {valid_acc:.2f} {test_acc:.2f}', loss=f'{train_loss:.4f} {valid_loss:.4f} {test_loss:.4f}')

    test_at_min_loss = test_ACCs[np.argmin(valid_LOSSs)]
    if verbose:
        print('-' * 80)
        print(f'Finished experiment {tag}')
    print(f'Accuracy: test_at_min_loss={test_at_min_loss:.2f}, max_test={max(test_ACCs):.2f}, ' +
          f'max_train={max(train_ACCs):.2f}, max_valid={max(valid_ACCs):.2f}')
    model.load_state_dict(torch.load(f'temp/models/{tag}_{timestamp}'))
    return model, test_ACCs[-1], max(test_ACCs), init_acc


def adapt_wrapper(model_cls, pretrain_model, n_activities=6, n_hidden=128, tag=None, baseline_mode=False, **kwargs):
    new_model = model_cls(n_activities=n_activities, n_hidden=n_hidden)
    tag = tag if tag else f'{model_cls.__name__}_adapt'

    if pretrain_model:
        new_model.load_state_dict(pretrain_model.state_dict())
        # for p in new_model.parameters():
        #     p.requires_grad = False
        # new_model.pos_weight.requires_grad = True
        # new_model.pos_emb.requires_grad = True
        # for p in new_model.pos_encoder.parameters():
        #     p.requires_grad = True
        # for p in new_model.client_encoder.parameters():
        #     p.requires_grad = True
        # for p in new_model.transformer_encoder.parameters():
        #     p.requires_grad = True
    if model_cls == MixRx_Transformer and not baseline_mode:
        return adapt_MixRx_Transformer(
            new_model, source_model=pretrain_model, device=device,
            tag=tag, **kwargs
        )
    else:
        return adapt_baseline(
            new_model, source_model=pretrain_model, device=device,
            tag=tag, **kwargs
        )


def cross_env_experiments():
    model_classes = [MixRx_Transformer, Client_Transformer, Client_Mean]
    # settings = ['adv', 'normal']
    settings = ['normal']
    split_ratio = 0.7
    n_shots_options = [1, 10, 20]
    verbose = False
    n_hidden = 128
    train_epochs = 600
    tune_epochs = 200
    env_configs = [
        {'target_room':0, 'target_user':0},
        {'target_room':1, 'target_user':1},
        {'target_room':2, 'target_user':2},
    ]
    results = []

    all_test_df, all_test_ds = load_data(rooms=None, users=None, split_head=False, split_ratio=split_ratio)
    assign_rx_pos = Assign_Rx_Pos([all_test_ds], granularity='room')

    for env_config in env_configs:
        target_room = env_config['target_room']
        target_user = env_config['target_user']
        source_rooms = list(set([0,1,2]) - set([target_room]))
        train_df, train_ds = load_data(rooms=source_rooms, users=None, split_head=True, split_ratio=split_ratio)
        valid_df, valid_ds = load_data(rooms=source_rooms, users=None, split_head=False, split_ratio=split_ratio)
        tune_df, tune_ds = load_data(rooms=[target_room], users=[target_user], split_head=True, split_ratio=0.5)
        test_df, test_ds = load_data(rooms=[target_room], users=[target_user], split_head=False, split_ratio=0.5)
        train_ds.transform = assign_rx_pos.augment_transform
        valid_ds.transform = assign_rx_pos.default_transform
        tune_ds.transform = assign_rx_pos.augment_transform
        test_ds.transform = assign_rx_pos.default_transform
        # train_ds.transform = assign_rx_pos.default_transform
        # valid_ds.transform = assign_rx_pos.default_transform
        # tune_ds.transform = assign_rx_pos.default_transform
        # test_ds.transform = assign_rx_pos.default_transform

        for model_cls in model_classes:
            model_name = model_cls if isinstance(model_cls, str) else model_cls.__name__
            tag = f'{model_name}-r{target_room}-u{target_user}-train'
            print('=' * 80)
            print(f'training model in source domain: {tag}')
            model, test_acc, test_acc_max, _ = adapt_wrapper(
                model_cls, None, tag=tag, verbose=verbose, n_epochs=train_epochs, loss_mode='source', lr=1e-3,
                train_ds=train_ds, valid_ds=valid_ds, tune_ds=tune_ds, test_ds=test_ds
            )

            for setting in settings:
                for n_shots in n_shots_options:
                    base_tag = f'{model_name}-r{target_room}-u{target_user}-{setting}-{n_shots}shot'

                    current_tune_ds = tune_ds if setting == 'normal' else tune_ds_adv
                    current_test_ds = test_ds if setting == 'normal' else test_ds_adv
                    print('-' * 80)
                    print(f'adapting model: {base_tag}')
                    current_tune_ds = get_n_shots_ds(current_tune_ds, n=n_shots)

                    if model_cls == MixRx_Transformer:
                        model_adapt, acc_adapt, acc_max_adapt, acc_init2 = adapt_wrapper(
                            model_cls, model, tag=f'{base_tag}-adapt', verbose=verbose,
                            baseline_mode=False, n_hidden=n_hidden, loss_mode='source+target+mix+inter',
                            n_epochs=tune_epochs, lr=1e-4, use_pos=True,
                            train_ds=train_ds, valid_ds=current_tune_ds, tune_ds=current_tune_ds, test_ds=current_test_ds,
                        )
                    else:
                        model_adapt, acc_adapt, acc_max_adapt, acc_init2 = None, None, None, None

                    print('fine-tuning')
                    model_finetune, acc_finetune, acc_max_finetune, acc_init = adapt_wrapper(
                        model_cls, model, tag=f'{base_tag}-finetune', verbose=verbose,
                        baseline_mode=True, n_hidden=n_hidden, loss_mode='source', n_epochs=tune_epochs, lr=1e-4,
                        train_ds=current_tune_ds, valid_ds=current_tune_ds, tune_ds=current_tune_ds, test_ds=current_test_ds,
                    )

                    print('co-training')
                    model_cotrain, acc_cotrain, acc_max_cotrain, acc_init = adapt_wrapper(
                        model_cls, model, tag=f'{base_tag}-cotrain', verbose=verbose,
                        baseline_mode=True, n_hidden=n_hidden, loss_mode='source+target', n_epochs=tune_epochs, lr=1e-4,
                        train_ds=train_ds, valid_ds=current_tune_ds, tune_ds=current_tune_ds, test_ds=current_test_ds
                    )

                    item = {
                        'model': model_name,
                        'target_room': target_room,
                        'target_user': target_user,
                        'original_acc': test_acc,
                        'setting': setting,
                        'n_shots': n_shots,
                        'acc_init': acc_init,
                        'acc_init2': acc_init2,
                        'acc_finetune': acc_finetune,
                        'acc_max_finetune': acc_max_finetune,
                        'acc_cotrain': acc_cotrain,
                        'acc_max_cotrain': acc_max_cotrain,
                        'acc_adapt': acc_adapt,
                        'acc_max_adapt': acc_max_adapt
                        # FIXME
                        # 'wight': model_adapt.pos_weight
                    }
                    print(item)
                    results.append(item)

    results_df = pd.DataFrame(results)
    results_df.to_csv('cross_room_results.csv')
    
    
def main():
    cross_env_experiments()
    
if __name__ == '__main__':
    main()
