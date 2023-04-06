import os
import math
import yaml
import torch
import argparse
from utils.load_dataset import background_appear_motion_dataset
from models.HSCModel import HSCModel
from models.MemoryBank import MemoryBank
from models.DecoderModel import DecoderModel
import torch.utils.data
from tqdm import tqdm
import numpy as np
from utils.utils import eval
from utils.utils import gaussian_filter
from utils.utils import min_max_norm
from utils.utils import calc_l2_error
from torch.nn import functional as F

def get_frame_scores(frames_scores, sign, feats_dict, video_name, feature_dim, temporature, memories, encoder, decoder):
    for clip_i in feats_dict[video_name].keys():
        error_clip = 0
        for cls_id in feats_dict[video_name][clip_i].keys():
            feats = torch.from_numpy(feats_dict[video_name][clip_i][cls_id]).cuda()
            encoder_output, _ = encoder(feats, sign=sign)
            similarity = torch.matmul(encoder_output / temporature, memories.T)
            similarity_softmax = F.softmax(similarity, dim=1)
            encoder_output = torch.matmul(similarity_softmax, memories)
            decoder_output = decoder(encoder_output, sign=sign)
            feats = feats[:, :feature_dim]
            l2_error = calc_l2_error(decoder_output, feats)
            error_clip = max(error_clip, torch.max(l2_error).cpu().item())
        frames_scores[clip_i * 16:(clip_i + 1) * 16] = error_clip

def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DC_Main')
    parser.add_argument('--dataset', type=str, default='ShanghaiTech', help="[ShanghaiTech/Avenue/UCSD_ped2]")
    parser.add_argument('--segment_len', type=int, default=16)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lambda_contrastive', type=float, default=0.2)
    parser.add_argument('--MB_momentum', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--temporature', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--appear_input_dim', type=int, default=2614)
    parser.add_argument('--motion_input_dim', type=int, default=2102)
    parser.add_argument('--appear_feature_dim', type=int, default=1024)
    parser.add_argument('--motion_feature_dim', type=int, default=512)
    parser.add_argument('--background_feature_dim', type=int, default=1590)
    parser.add_argument('--encoder_output_dim', type=int, default=1280)
    parser.add_argument('--num_cluster', type=int, default=13)
    parser.add_argument('--model_weight_init', action="store_true", help="init the model")
    parser.add_argument('--removeClusterHead', action="store_true", help="remove the cluster head")
    parser.add_argument('--removeMotionBranch', action="store_true", help="remove the motion branch")
    parser.add_argument('--appear_hidden_dim', type=int, default=1152)
    parser.add_argument('--motion_hidden_dim', type=int, default=1024)
    parser.add_argument('--train_dataset_path', type=str, default="")
    parser.add_argument('--test_dataset_path', type=str, default="")
    parser.add_argument('--test_mask_dir', type=str, default="")
    args = parser.parse_args()
    return args

def main(args):

    with open('config/'+args.dataset+'.yaml', encoding='utf-8') as config_file:
        data = yaml.load(config_file, Loader=yaml.FullLoader)

    appear_input_dim = data['appear_input_dim']
    motion_input_dim = data['motion_input_dim']
    background_feature_dim = data['background_feature_dim']
    encoder_output_dim = data['encoder_output_dim']
    appear_feature_dim = data['appear_feature_dim']
    motion_feature_dim = data['motion_feature_dim']
    temporature = data['temporature']
    num_cluster = data['num_cluster']
    removeMotionBranch = data['removeMotionBranch']
    augmentMotion = data['augmentMotion']
    train_dataset_path = data['train_dataset_path']
    test_dataset_path = data['test_dataset_path']
    test_mask_dir = data['test_mask_dir']
    MB_model_path = data['MB_model_path']
    encoder_model_path = data['encoder_model_path']
    decoder_model_path = data['decoder_model_path']
    segment_len = data['segment_len']

    filter_2d = gaussian_filter(np.arange(1, data['gaussian_len']), data['gaussian_sigma'])
    padding_size = len(filter_2d) // 2

    train_dataset = background_appear_motion_dataset(train_dataset_path, num_cluster)

    encoder = HSCModel(appear_input_dim=appear_input_dim, motion_input_dim=motion_input_dim,
                             output_dim=encoder_output_dim, num_cluster=num_cluster)
    state_dict = torch.load(encoder_model_path)
    encoder.load_state_dict(state_dict, False)
    encoder = encoder.cuda().eval()
    decoder = DecoderModel(input_dim=encoder_output_dim, appear_output_dim=appear_feature_dim,
                           motion_output_dim=motion_feature_dim, background_output_dim=background_feature_dim,
                           hidden_dim=[1152, 1024])
    state_dict = torch.load(decoder_model_path)
    decoder.load_state_dict(state_dict, False)
    decoder = decoder.cuda().eval()

    MB = MemoryBank(num_feats=len(train_dataset), feats_dim=encoder_output_dim, momentum=0.9,
                    dataset_path=train_dataset_path)
    state_dict = torch.load(MB_model_path)
    MB.load_state_dict(state_dict, False)
    MB = MB.cuda().eval()

    test_dataset = np.load(test_dataset_path, allow_pickle=True).tolist()
    appear_feats_dict = test_dataset['appear_feats']
    motion_feats_dict = test_dataset['motion_feats']

    appear_memories = MB.get_all_elements(sign=0)
    motion_memories = MB.get_all_elements(sign=1)

    appear_scores_list = []
    motion_scores_list = []
    label_list = []

    assert len(appear_feats_dict.keys()) == len(motion_feats_dict.keys())

    with torch.no_grad():
        for video_name in tqdm(appear_feats_dict.keys()):
            gt_path = os.path.join(test_mask_dir, video_name+".npy")
            if os.path.exists(gt_path):
                gt = np.load(gt_path, allow_pickle=True)
            n_clips = gt.shape[0] // segment_len
            appear_frames_scores = np.zeros((n_clips * segment_len), dtype=np.float32)
            motion_frames_scores = np.zeros((n_clips * segment_len), dtype=np.float32)

            get_frame_scores(appear_frames_scores, 0, appear_feats_dict, video_name, appear_feature_dim, temporature,
                             appear_memories, encoder, decoder)
            appear_scores_list.extend(appear_frames_scores.tolist())
            get_frame_scores(motion_frames_scores, 1, motion_feats_dict, video_name, motion_feature_dim, temporature,
                             motion_memories, encoder, decoder)
            motion_scores_list.extend(motion_frames_scores.tolist())
            label_list.extend(gt[:n_clips*segment_len])

    appear_scores = min_max_norm(appear_scores_list)
    temp = np.concatenate((appear_scores[:padding_size], appear_scores, appear_scores[-padding_size:]))
    appear_scores = np.correlate(temp, filter_2d, 'valid')
    motion_scores = np.array(motion_scores_list)
    if removeMotionBranch == False:
        motion_scores = min_max_norm(motion_scores_list)
        temp = np.concatenate((motion_scores[:padding_size], motion_scores, motion_scores[-padding_size:]))
        motion_scores = np.correlate(temp, filter_2d, 'valid')
    if augmentMotion == True:
        augment_motion_path = data['augment_motion_path']
        motion_scores = np.load(augment_motion_path, allow_pickle=True).tolist()
        motion_scores = np.array(motion_scores)
    frame_scores = appear_scores * 0.5 + motion_scores * 0.5
    frame_scores_list = frame_scores.tolist()
    auc_test = eval(frame_scores_list, label_list, None)
    print ("auc = ", auc_test)


if __name__ == '__main__':
    args = parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)