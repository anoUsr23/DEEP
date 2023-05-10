from __future__ import print_function
import argparse
import pandas as pd
import os
import os.path as osp
import copy
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataloader import LLP_dataset, ToTensor, categories
from nets.net_audiovisual import DEEP, LabelSmoothingNCELoss
from utils.eval_metrics import segment_level, event_level, print_overall_metric

from utils.logger import get_logger
from nets.criterion import EvidenceLoss
from tqdm import tqdm
import datetime
import pickle as pkl
import pdb

exp_logger = get_logger()


def get_LLP_dataloader(args):
    train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir,
                                video_dir=args.video_dir, st_dir=args.st_dir,
                                transform=transforms.Compose([ToTensor()]),
                                a_smooth=args.a_smooth, v_smooth=args.v_smooth, mode=args.mode)
    val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir,
                              video_dir=args.video_dir, st_dir=args.st_dir,
                              transform=transforms.Compose([ToTensor()]), mode=args.mode)
    test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir,
                               video_dir=args.video_dir, st_dir=args.st_dir,
                               transform=transforms.Compose([ToTensor()]), mode=args.mode)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=5, pin_memory=True, sampler=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=1, pin_memory=True, sampler=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True, sampler=None)

    return train_loader, val_loader, test_loader



def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_random_state():
    state = {
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(),
        'random_rng': random.getstate(),
        'numpy_rng': np.random.get_state()
    }
    return state



def train_with_label_smoothing(args, model, train_loader, optimizer, criterion, epoch, logger):
    exp_logger(f"begin train_with_label_smoothing.")
    model.train()

    criterion = LabelSmoothingNCELoss(classes=10, smoothing=args.nce_smooth)

    criterion_edl_modality = EvidenceLoss(num_classes=2, evidence = 'exp')

    for batch_idx, sample in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training with label smoothing epoch {}".format(epoch)):
        audio, video, video_st, target = sample['audio'].to('cuda'), \
                                         sample['video_s'].to('cuda'), \
                                         sample['video_st'].to('cuda'), \
                                         sample['label'].type(torch.FloatTensor).to('cuda')
        Pa, Pv = sample['Pa'].type(torch.FloatTensor).to('cuda'), sample['Pv'].type(torch.FloatTensor).to('cuda')


        optimizer.zero_grad()
        global_logits, a_logits, v_logits, frame_logits, temporal_logits, frame_att, temporal_att, sims_after, mask_after = model(audio, video, video_st, with_ca=True, epoch_num = epoch-1)


        loss1 = criterion(sims_after, mask_after) 
        
        global_sl_loss = criterion_edl_modality(global_logits.reshape(-1, 2), torch.stack((target, 1-target), dim=-1).reshape(-1, 2))['loss_cls'].sum(dim=-1).mean()
        frame_target = target.unsqueeze(dim=1).expand(-1, 10, 25).reshape(-1)
        temporal_sl_loss = criterion_edl_modality(temporal_logits.reshape(-1, 2), torch.stack((frame_target, 1-frame_target)).reshape(-1, 2))['loss_cls'].reshape(-1, 25).sum(dim=-1).mean()
        loss2 = temporal_sl_loss + global_sl_loss

        audio_sl_loss = criterion_edl_modality(a_logits.reshape(-1, 2), torch.stack((Pa, 1-Pa), dim=-1).reshape(-1, 2))['loss_cls'].reshape(-1, 25).sum(dim=-1).mean()
        video_sl_loss = criterion_edl_modality(v_logits.reshape(-1, 2), torch.stack((Pv, 1-Pv), dim=-1).reshape(-1, 2))['loss_cls'].reshape(-1, 25).sum(dim=-1).mean()

        loss3 = audio_sl_loss + video_sl_loss 


        loss4 = torch.nn.functional.cross_entropy(frame_att[..., 0].mean(dim=-2).permute(0, 2, 1).reshape(-1, 10), temporal_att.permute(0, 2, 1).reshape(-1, 10)).mean()
        

        loss = loss1 + loss2 + loss3 + loss4

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:

            log_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.3f}\tLoss2: {:.3f}\tLoss3: {:.3f}\tLoss4: {:.3f}'.format(epoch, batch_idx * len(audio), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss1.item(), loss2.item(), loss3.item(), loss4.item())
            exp_logger(log_str)



def train_with_label_refinement(args, models, train_loader, optimizers, criterion, epoch, logger):
    ##########################################################
    #    Complete Code will be released if paper accepted    #
    ##########################################################
    pass


def eval(args, model, val_loader, set, epoch = 0):
    model.eval()
    exp_logger("begin evaluate.")
    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv("data/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("data/AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():

        criterion_edl_modality = EvidenceLoss(num_classes=2, evidence = 'exp')
        for batch_idx, sample in tqdm(enumerate(val_loader), total=len(val_loader), desc="Eval epoch {}".format(epoch)):
            audio, video, video_st, target = sample['audio'].to('cuda'), \
                                             sample['video_s'].to('cuda'), \
                                             sample['video_st'].to('cuda'), \
                                             sample['label'].to('cuda')
            

            _, a_logits, v_logits, frame_logits, _, _, _, _, _ = model(audio, video, video_st)

            a_sl_prob, _ = criterion_edl_modality.get_predictions(a_logits)
            v_sl_prob, _ = criterion_edl_modality.get_predictions(v_logits)
            
            frame_sl_prob, _ = criterion_edl_modality.get_predictions(frame_logits)

            a_prob = torch.clamp(a_sl_prob[:,:,0], min=1e-7, max=1 - 1e-7)
            v_prob = torch.clamp(v_sl_prob[:,:,0], min=1e-7, max=1 - 1e-7)
            a_temporal_prob = frame_sl_prob[:,:,0,:, 0]
            v_temporal_prob = frame_sl_prob[:,:,1,:, 0]


            Pa = a_temporal_prob.cpu().detach().numpy()[0, :, :]
            Pv = v_temporal_prob.cpu().detach().numpy()[0, :, :]
            a_prob = a_prob.cpu().detach().numpy()
            v_prob = v_prob.cpu().detach().numpy()
            a_sl_prob = a_sl_prob.cpu().detach().numpy()
            v_sl_prob = v_sl_prob.cpu().detach().numpy()


            a_thresh_list = np.array([[0.30, 0.30, 0.30, 0.40, 0.30, 0.70, 0.70, 0.70, 0.70, 0.30, 0.20, 0.30, 0.20, 0.40, 0.40, 0.40, 0.60, 0.8, 0.50, 0.60, 0.30, 0.40, 0.20, 0.50, 0.20]])
            v_thresh_list = np.array([[0.95, 0.20, 0.50, 0.55, 0.20, 0.85, 0.45, 0.65, 0.45, 0.55, 0.25, 0.30, 0.35, 0.55, 0.20, 0.20, 0.25, 0.30, 0.40, 0.35, 0.15, 0.50, 0.55, 0.50, 0.30]])

            oa = (a_prob >= a_thresh_list).astype(np.int_)
            ov = (v_prob >= v_thresh_list).astype(np.int_)

            # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.25).astype(np.int_) * np.repeat(oa, repeats=10, axis=0)
            Pv = (Pv >= 0.25).astype(np.int_) * np.repeat(ov, repeats=10, axis=0)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v = np.zeros((25, 10))


            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            # extract visual GT labels
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1
    
            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)



    audio_segment_level, visual_segment_level, av_segment_level, avg_type, avg_event, \
        audio_event_level, visual_event_level, av_event_level, avg_type_event, avg_event_level \
        = print_overall_metric(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av)
    return audio_segment_level, visual_segment_level, av_segment_level, avg_type, avg_event, \
        audio_event_level, visual_event_level, av_event_level, avg_type_event, avg_event_level


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument("--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
    parser.add_argument("--video_dir", type=str, default='data/feats/res152/', help="video dir")
    parser.add_argument("--st_dir", type=str, default='data/feats/r2plus1d_18/', help="video dir")
    parser.add_argument("--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument("--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument("--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--warm_up_epoch', type=float, default=0.9, help='warm-up epochs')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_step_size', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=6, help='random seed')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument("--mode", type=str, default='train_with_label_smoothing',
                        choices=['train_with_label_smoothing', 'train_with_label_refinement', 'test_model'],
                        help="with mode to use")
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--noise_ratio_file', type=str, default='noise_ratios.npz')
    parser.add_argument('--a_smooth', type=float, default=1.0)
    parser.add_argument('--v_smooth', type=float, default=0.9)
    parser.add_argument('--clamp', type=float, default=1e-7)
    parser.add_argument('--nce_smooth', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.2, help='feature temperature number')
    parser.add_argument('--warm_num', type=int, default=1000, help='warm epoch number')
    parser.add_argument('--log_interval', type=int, default=700, help='how many batches for logging training status')
    parser.add_argument('--log_file', type=str, help="log file path")
    parser.add_argument('--save_model', type=str, default="true", choices=["true", "false"], help='whether to save model')
    parser.add_argument("--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument("--checkpoint", type=str, default='DEEP_LS', help="save model name")

    parser.add_argument("--logger", type=str, default='logger/', help="save model name")
    parser.add_argument("--best_metric", type=str, default='avg_type_seg', choices=["audio_seg", "visual_seg", "av_seg", "avg_type_seg", "avg_event_seg", "audio_eve", "visual_eve", "av_eve", "avg_type_eve", "avg_event_eve"], help="best metric type")
    parser.add_argument("--no-log", action='store_true', default=False, help="logger switcher")
    args = parser.parse_args()

    if args.no_log:
        exp_logger.disable_file()
    else:
        os.makedirs(os.path.dirname(args.logger), exist_ok=True)
        logger_timestamps = datetime.datetime.now()
        exp_logger.set_file(os.path.join(args.logger, args.mode+'_'+datetime.datetime.strftime(logger_timestamps,'%Y-%m-%d %H:%M:%S')+".log"))

    save_dir = osp.join(args.model_save_dir, args.checkpoint)
    os.makedirs(save_dir, exist_ok=True)

    # print parameters
    exp_logger('----------------args-----------------')
    for k in list(vars(args).keys()):
        exp_logger('%s: %s' % (k, vars(args)[k]))
    exp_logger('----------------args-----------------')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    exp_logger(f'current time: {cur}')

    set_random_seed(args.seed)
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    save_model = args.save_model == 'true'
    os.makedirs(args.model_save_dir, exist_ok=True)

    model = DEEP(args.num_layers, args.temperature, warm_num = args.warm_num).to('cuda')

    start = time.time()

    if args.mode == 'train_with_label_smoothing':
        logger = SummaryWriter(args.log_file) if args.log_file else None

        args.with_ca = False
        train_loader, val_loader, test_loader = get_LLP_dataloader(args)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        criterion = nn.BCELoss()

        best_F = 0
        best_model = None
        for epoch in range(1, args.epochs + 1):
            train_with_label_smoothing(args, model, train_loader, optimizer, criterion, epoch=epoch, logger=logger)
            scheduler.step(epoch)

            exp_logger("Validation Performance of Epoch {}:".format(epoch))
            
            audio_seg, visual_seg, av_seg, avg_type_seg, avg_event_seg, audio_eve, visual_eve, av_eve, avg_type_eve, avg_event_eve = eval(args, model, val_loader, args.label_val, epoch)


            state_dict = get_random_state()
            state_dict['model'] = model.state_dict()
            state_dict['optimizer'] = optimizer.state_dict()
            state_dict['scheduler'] = scheduler.state_dict()
            state_dict['epochs'] = args.epochs

            os.makedirs(os.path.dirname(osp.join(args.model_save_dir, args.checkpoint)), exist_ok=True)
            save_path = osp.join(args.model_save_dir, args.checkpoint, args.checkpoint+'.ckpt')
            exp_logger("Saving checkpoint at epoch {}: {} \n".format(epoch, save_path))
            torch.save(state_dict, save_path)

            if locals()[args.best_metric] >= best_F:
                best_F = locals()[args.best_metric]
                best_model = copy.deepcopy(model)
                if save_model:
                    save_path = osp.join(args.model_save_dir, args.checkpoint, args.checkpoint+'.best')
                    torch.save(state_dict, save_path)
                
                    exp_logger("Best results at {} have been updated: {} \n".format(args.best_metric, save_path))

        if logger:
            logger.close()
        optimizer.zero_grad()
        model = best_model
        exp_logger("Test the best model:")
        eval(args, model, test_loader, args.label_test)

    elif args.mode == 'train_with_label_refinement':
        logger = SummaryWriter(args.log_file) if args.log_file else None
        args.with_ca = True

        train_loader, val_loader, test_loader = get_LLP_dataloader(args)
        exp_logger("Complete Code will be released if paper accepted !!")

        ##########################################################
        #    Complete Code will be released if paper accepted    #
        ##########################################################

    elif args.mode == 'test_models':
        dataset = args.label_test
        args.with_ca = True 
        test_dataset = LLP_dataset(label=dataset, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir,
                                   transform=transforms.Compose([ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        resume_path = osp.join(args.model_save_dir, args.checkpoint, args.checkpoint+'.best')
        exp_logger("Resuming models from {}".format(resume_path))
        resume = torch.load(resume_path)
        model.load_state_dict(resume['model'])
        eval(args, model, test_loader, dataset)

    end = time.time()
    exp_logger(f'duration time {(end - start) / 60} mins.')
    cur = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    exp_logger(f'current time: {cur}')


if __name__ == '__main__':
    main()
