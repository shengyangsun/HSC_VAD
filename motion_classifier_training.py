import os
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from utils.load_dataset import regressor_dataset
from models.HSCModel import HSCModel
from models.MemoryBank import MemoryBank
from models.DecoderModel import DecoderModel
from models.Regressor import Regressor
from utils.utils import get_CE_loss, set_seeds, gaussian_filter, eval, calc_l2_error, min_max_norm, get_n_clips
from tqdm import tqdm

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DC_Main')
    parser.add_argument('--dataset', type=str, default='ShanghaiTech', help="[ShanghaiTech/Avenue/UCSD_ped2]")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--inter_epoch', type=int, default=1)

    parser.add_argument('--lambda_contrastive', type=float, default=0.2)
    parser.add_argument('--MB_momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--temporature', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--appear_input_dim', type=int, default=2614)
    parser.add_argument('--motion_input_dim', type=int, default=2102)
    parser.add_argument('--appear_feature_dim', type=int, default=1024)
    parser.add_argument('--motion_feature_dim', type=int, default=512)
    parser.add_argument('--background_feature_dim', type=int, default=1590)
    parser.add_argument('--encoder_output_dim', type=int, default=1280)
    parser.add_argument('--num_cluster', type=int, default=23)
    parser.add_argument('--model_weight_init', action="store_true", help="init the model")


    parser.add_argument('--train_dataset_path', type=str, default="")
    parser.add_argument('--test_dataset_path', type=str, default= "")
    parser.add_argument('--test_mask_dir', type=str, default="")
    parser.add_argument('--model_saved_dir', type=str, default="")

    args = parser.parse_args()
    return args


def get_CE_loss(positive_outputs, negative_outputs):
    loss = torch.mean(-torch.log(positive_outputs+1e-10)-torch.log(1-negative_outputs+1e-10))
    return loss


def train(args):

    with open('config/'+args.dataset+'.yaml', encoding='utf-8') as config_file:
        data = yaml.load(config_file, Loader=yaml.FullLoader)

    args.test_dataset_path = data['test_dataset_path']
    args.test_mask_dir = data['test_mask_dir']
    gaussian_len = data['gaussian_len']
    gaussian_sigma = data['gaussian_sigma']
    n_clips_dict = get_n_clips(args.dataset)

    train_dataset = regressor_dataset(args.train_dataset_path)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    test_dataset = np.load(args.test_dataset_path, allow_pickle=True).tolist()
    appear_feats_dict = test_dataset['appear_feats']
    motion_feats_dict = test_dataset['motion_feats']
    filter_2d = gaussian_filter(np.arange(1, gaussian_len), gaussian_sigma)
    padding_size = len(filter_2d) // 2

    model = Regressor(input_feature_dim=args.motion_input_dim)
    model = model.cuda().train()

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}],
                                 weight_decay=args.weight_decay)
    MSELoss = torch.nn.MSELoss()

    best_test_auc = 0

    for epoch in range(args.epochs):
        iter_count = 0
        for positive_feats, negative_feats in dataloader:
            positive_feats = positive_feats.view([-1, args.motion_input_dim]).cuda().float()
            negative_feats = negative_feats.view([-1, args.motion_input_dim]).cuda().float()

            positive_outputs = model(positive_feats)
            negative_outputs = model(negative_feats)
            loss = get_CE_loss(positive_outputs, negative_outputs)
            print('[{}/{}]: loss {:.4f}, '.format(iter_count, epoch, loss))
            iter_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.inter_epoch == 0:
            appear_scores_list = []
            motion_scores_list = []
            label_list = []
            model = model.eval()
            with torch.no_grad():
                for video_name in tqdm(appear_feats_dict.keys()):
                    n_clips = n_clips_dict[video_name]
                    gt_path = os.path.join(args.test_mask_dir, video_name + ".npy")
                    if os.path.exists(gt_path):
                        gt = np.load(gt_path, allow_pickle=True)
                    else:
                        gt = np.zeros((n_clips * args.segment_len), dtype=np.float32)
                    motion_frames_scores = np.zeros((n_clips * args.segment_len), dtype=np.float32)

                    scores_list = []
                    for clip_i in motion_feats_dict[video_name].keys():
                        error_clip = 0
                        for cls_id in motion_feats_dict[video_name][clip_i].keys():
                            feats = torch.from_numpy(motion_feats_dict[video_name][clip_i][cls_id]).cuda()
                            score_output = model(feats)
                            error_clip = max(error_clip, torch.max(score_output).cpu().item())
                        motion_frames_scores[clip_i * args.segment_len:(clip_i + 1) * args.segment_len] = error_clip
                    motion_scores_list.extend(motion_frames_scores.tolist())
                    label_list.extend(gt[:n_clips * args.segment_len])

            motion_scores = np.array(motion_scores_list)
            temp = np.concatenate((motion_scores[:padding_size], motion_scores, motion_scores[-padding_size:]))
            motion_scores = np.correlate(temp, filter_2d, 'valid')
            frame_scores_list = motion_scores.tolist()

            auc_test = eval(frame_scores_list, label_list, None)
            print("test auc = ", auc_test)
            if auc_test > best_test_auc:
                best_test_auc = auc_test
                torch.save(model.state_dict(), os.path.join(args.model_saved_dir, args.tip+"MA_epoch_" + str(epoch) + "_" + str(best_test_auc)))
                print ("saved finished.")
                np.save("/data/ssy/code/HSC_VAD/data/"+args.dataset+"_MA.npy", frame_scores_list)
            model = model.train()
        dataloader.dataset.shuffle_keys()

if __name__ == '__main__':
    args = parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    train(args)