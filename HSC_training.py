import os
import yaml
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from utils.load_dataset import background_appear_motion_dataset
from models.HSCModel import HSCModel
from models.MemoryBank import MemoryBank
from models.DecoderModel import DecoderModel
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
    parser.add_argument('--num_cluster', type=int, default=13)
    parser.add_argument('--model_weight_init', action="store_true", help="init the model")
    parser.add_argument('--removeClusterHead', action="store_true", help="remove the cluster head")
    parser.add_argument('--removeMotionBranch', action="store_true", help="remove the motion branch")
    parser.add_argument('--appear_hidden_dim', type=int, default=1152)
    parser.add_argument('--motion_hidden_dim', type=int, default=1024)
    parser.add_argument('--intra_lambda', type=float, default=1)
    parser.add_argument('--inter_lambda', type=float, default=1)
    parser.add_argument('--LC_lambda', type=float, default=1)
    parser.add_argument('--train_dataset_path', type=str, default="")
    parser.add_argument('--test_dataset_path', type=str, default="")
    parser.add_argument('--test_mask_dir', type=str, default="")
    parser.add_argument('--model_saved_dir', type=str, default= \
        "/data/ssy/code/HSC_VAD/saved_models/")

    args = parser.parse_args()
    return args


def train(args):

    with open('config/'+args.dataset+'.yaml', encoding='utf-8') as config_file:
        data = yaml.load(config_file, Loader=yaml.FullLoader)

    args.appear_input_dim = data['appear_input_dim']
    args.motion_input_dim = data['motion_input_dim']
    args.background_feature_dim = data['background_feature_dim']
    args.encoder_output_dim = data['encoder_output_dim']
    args.appear_feature_dim = data['appear_feature_dim']
    args.motion_feature_dim = data['motion_feature_dim']
    args.temporature = data['temporature']
    args.num_cluster = data['num_cluster']
    args.batch_size = data['batch_size']
    args.removeMotionBranch = data['removeMotionBranch']
    args.removeClusterHead = data['removeClusterHead']
    args.train_dataset_path = data['train_dataset_path']
    args.test_dataset_path = data['test_dataset_path']
    args.test_mask_dir = data['test_mask_dir']
    gaussian_len = data['gaussian_len']
    gaussian_sigma = data['gaussian_sigma']

    n_clips_dict = get_n_clips(args.dataset)

    train_dataset = background_appear_motion_dataset(args.train_dataset_path, args.num_cluster, removeMotionBranch=args.removeMotionBranch)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    test_dataset = np.load(args.test_dataset_path, allow_pickle=True).tolist()
    appear_feats_dict = test_dataset['appear_feats']
    if args.removeMotionBranch == False:
        motion_feats_dict = test_dataset['motion_feats']

    filter_2d = gaussian_filter(np.arange(1, gaussian_len), gaussian_sigma)
    padding_size = len(filter_2d) // 2

    model = HSCModel(appear_input_dim=args.appear_input_dim, motion_input_dim=args.motion_input_dim,
                             output_dim=args.encoder_output_dim, num_cluster=args.num_cluster,
                             removeClusterHead=args.removeClusterHead, weight_init=args.model_weight_init)
    model = model.cuda().train()
    MB = MemoryBank(num_feats=len(train_dataset), feats_dim=args.encoder_output_dim, momentum=args.MB_momentum, dataset_path=args.train_dataset_path)
    decoder = DecoderModel(input_dim=args.encoder_output_dim, appear_output_dim=args.appear_feature_dim,
                           motion_output_dim=args.motion_feature_dim, background_output_dim=args.background_feature_dim,
                           hidden_dim=[1152, 1024])
    decoder = decoder.cuda().train()

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr},
                                  {"params": decoder.parameters(), "lr": args.lr}],
                                 weight_decay=args.weight_decay)
    MSELoss = torch.nn.MSELoss()
    best_test_auc = 0
    for epoch in range(args.epochs):
        iter_count = 0
        for feats, labels, back_ids, cls_ids, bank_ids, signs in dataloader:
            feats = feats.cuda()
            labels = labels.cuda().view([args.batch_size, -1])

            appear_decoder_loss = 0.0
            motion_decoder_loss = 0.0

            appear_index = torch.nonzero(signs==0).view([-1]) #sign_0: appear
            if appear_index.shape[0] > 0:
                appear_feats = feats[appear_index, :args.appear_input_dim]
                appear_labels = labels[appear_index].view([-1, args.num_cluster])
                appear_back_ids = back_ids[appear_index].view([-1])
                appear_cls_ids = cls_ids[appear_index].view([-1])
                appear_bank_ids = bank_ids[appear_index].view([-1])
                if args.removeClusterHead == False:
                    appear_encoder_outputs, appear_cluster_outputs = model(inputs=appear_feats, sign=0)
                else:
                    appear_encoder_outputs = model(inputs=appear_feats, sign=0)

                appear_decoder_outputs = decoder(appear_encoder_outputs, sign=0)
                appear_decoder_loss = MSELoss(appear_decoder_outputs, appear_feats[:, :args.appear_feature_dim])

            motion_index = np.array([])
            if args.removeMotionBranch == False:
                motion_index = torch.nonzero(signs == 1).view([-1])  # sign_1: motion
                if motion_index.shape[0] > 0:
                    motion_feats = feats[motion_index, :args.motion_input_dim]  # align the dimensionality
                    motion_labels = labels[motion_index].view([-1, args.num_cluster])
                    motion_back_ids = back_ids[motion_index].view([-1])
                    motion_cls_ids = cls_ids[motion_index].view([-1])
                    motion_bank_ids = bank_ids[motion_index].view([-1])
                    if args.removeClusterHead == False:
                        motion_encoder_outputs, motion_cluster_outputs = model(inputs=motion_feats, sign=1)
                    else:
                        motion_encoder_outputs = model(inputs=motion_feats, sign=1)
                    motion_decoder_outputs = decoder(motion_encoder_outputs, sign=1)
                    motion_decoder_loss = MSELoss(motion_decoder_outputs, motion_feats[:, :args.motion_feature_dim])

            if args.removeClusterHead == False:
                appear_cluster_CE_loss = get_CE_loss(appear_cluster_outputs, appear_labels)
                if args.removeMotionBranch == False:
                    motion_cluster_CE_loss = get_CE_loss(motion_cluster_outputs, motion_labels)
            else:
                appear_cluster_CE_loss = 0
                motion_cluster_CE_loss = 0

            #contrastive loss
            appear_inter_contrastive_loss, motion_inter_contrastive_loss = 0.0, 0.0
            appear_intra_contrastive_loss, motion_intra_contrastive_loss = 0.0, 0.0
            with torch.no_grad():
                appear_all_elems = MB.get_all_elements(sign=0).cuda()
                if args.removeMotionBranch == False:
                    motion_all_elems = MB.get_all_elements(sign=1).cuda()
            if epoch > 0:
                for back_id in range(args.num_cluster):
                    #back_id += 1
                    for sign_i in range(2): #0:appearance 1:motion
                        if (sign_i == 1) and (args.removeMotionBranch == True):
                            continue
                        if sign_i == 0:
                            back_indexes_batch = torch.nonzero(appear_back_ids == back_id).view([-1])
                        else:
                            back_indexes_batch = torch.nonzero(motion_back_ids == back_id).view([-1])
                        if back_indexes_batch.shape[0] > 0:
                            # indexes of same back_id
                            back_indexes_bank = dataloader.dataset.get_back_indexes(back_id=back_id, sign=sign_i)
                            with torch.no_grad():
                                elems_back = MB.get_elements(back_indexes_bank).cuda()  #(M, dim)
                            if sign_i == 0:
                                feats = appear_encoder_outputs[back_indexes_batch] #feats in batch
                            else:
                                feats = motion_encoder_outputs[back_indexes_batch]
                            similarity_matrix_back = torch.matmul(feats, elems_back.T)
                            exp_similarity_matrix_back = torch.exp(similarity_matrix_back / args.temporature)
                            if sign_i == 0:
                                all_similarity_matrix = torch.matmul(feats, appear_all_elems.T)
                            else:
                                all_similarity_matrix = torch.matmul(feats, motion_all_elems.T)
                            exp_all_similarity_matrix = torch.exp(all_similarity_matrix / args.temporature)

                            for ind_i in range(back_indexes_batch.shape[0]):
                                if sign_i == 0:
                                    cls_id = appear_cls_ids[back_indexes_batch[ind_i]].item()
                                else:
                                    cls_id = motion_cls_ids[back_indexes_batch[ind_i]].item()
                                cls_in_back_indexes = dataloader.dataset.get_index_cls_in_backid(back_id=back_id, cls_id=cls_id, sign=sign_i)
                                #return relative position
                                back_in_all_indexes = dataloader.dataset.get_indexes_back_in_all(back_id=back_id, sign=sign_i)
                                #intra loss (in back)
                                numerator = torch.sum(exp_similarity_matrix_back[ind_i, cls_in_back_indexes])
                                nominator = torch.sum(exp_similarity_matrix_back[ind_i, :])
                                if sign_i == 0:
                                    appear_intra_contrastive_loss = appear_intra_contrastive_loss - torch.log(numerator / nominator)
                                else:
                                    motion_intra_contrastive_loss = motion_intra_contrastive_loss - torch.log(numerator / nominator)
                                #inter loss (all)
                                numerator = torch.sum(exp_all_similarity_matrix[ind_i, back_in_all_indexes])
                                nominator = torch.sum(exp_all_similarity_matrix[ind_i, :])
                                if sign_i == 0:
                                    appear_inter_contrastive_loss = appear_inter_contrastive_loss - torch.log(numerator / nominator)
                                else:
                                    motion_inter_contrastive_loss = motion_inter_contrastive_loss - torch.log(numerator / nominator)

                if appear_index.shape[0] > 0:
                    appear_intra_contrastive_loss = appear_intra_contrastive_loss / appear_index.shape[0]
                    appear_inter_contrastive_loss = appear_inter_contrastive_loss / appear_index.shape[0]
                if motion_index.shape[0] > 0:
                    motion_intra_contrastive_loss = motion_intra_contrastive_loss / motion_index.shape[0]
                    motion_inter_contrastive_loss = motion_inter_contrastive_loss / motion_index.shape[0]

            loss = args.intra_lambda*appear_intra_contrastive_loss + args.inter_lambda*appear_inter_contrastive_loss + args.LC_lambda*appear_cluster_CE_loss + appear_decoder_loss + \
                   args.intra_lambda*motion_intra_contrastive_loss + args.inter_lambda*motion_inter_contrastive_loss + args.LC_lambda*motion_cluster_CE_loss + motion_decoder_loss
            print('[{}/{}]: loss {:.4f}, appear_intra_contrastive_loss {:.4f}, appear_inter_contrastive_loss {:.4f}, '
                  'appear_cluster_loss {:.4f}, appear_decoder_loss {:.4f}, \nmotion_intra_contrastive_loss {:.4f}, motion_inter_contrastive_loss {:.4f}, '
                  'motion_cluster_loss {:.4f}, motion_decoder_loss {:.4f}'.format(iter_count, epoch, loss, appear_intra_contrastive_loss, appear_inter_contrastive_loss, appear_cluster_CE_loss, appear_decoder_loss,
                                                       motion_intra_contrastive_loss, motion_inter_contrastive_loss, motion_cluster_CE_loss, motion_decoder_loss))
            iter_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if appear_index.shape[0] > 0:
                    MB.update(appear_encoder_outputs, appear_bank_ids)
                if motion_index.shape[0] > 0:
                    MB.update(motion_encoder_outputs, motion_bank_ids)

        if epoch % args.inter_epoch == 0:
            appear_scores_list = []
            motion_scores_list = []
            label_list = []
            model = model.eval()
            decoder = decoder.eval()
            with torch.no_grad():
                appear_memories = MB.get_all_elements(sign=0)
                if args.removeMotionBranch == False:
                    motion_memories = MB.get_all_elements(sign=1)
                for video_name in tqdm(appear_feats_dict.keys()):
                    n_clips = n_clips_dict[video_name]
                    gt_path = os.path.join(args.test_mask_dir, video_name + ".npy")
                    if os.path.exists(gt_path):
                        gt = np.load(gt_path, allow_pickle=True)
                    else:
                        gt = np.zeros((n_clips * args.segment_len), dtype=np.float32)
                    appear_frames_scores = np.zeros((n_clips * args.segment_len), dtype=np.float32)
                    motion_frames_scores = np.zeros((n_clips * args.segment_len), dtype=np.float32)

                    scores_list = []
                    for clip_i in appear_feats_dict[video_name].keys():
                        error_clip = 0
                        for cls_id in appear_feats_dict[video_name][clip_i].keys():
                            feats = torch.from_numpy(appear_feats_dict[video_name][clip_i][cls_id]).cuda()
                            if args.removeClusterHead == False:
                                encoder_output, _ = model(feats, sign=0)
                            else:
                                encoder_output = model(feats, sign=0)
                            similarity = torch.matmul(encoder_output / args.temporature, appear_memories.T)
                            similarity_softmax = F.softmax(similarity, dim=1)
                            encoder_output = torch.matmul(similarity_softmax, appear_memories)
                            decoder_output = decoder(encoder_output, sign=0)
                            appear_feats = feats[:, :args.appear_feature_dim]
                            l2_error = calc_l2_error(decoder_output, appear_feats)
                            error_clip = max(error_clip, torch.max(l2_error).cpu().item())
                        appear_frames_scores[clip_i * args.segment_len:(clip_i + 1) * args.segment_len] = error_clip
                    appear_scores_list.extend(appear_frames_scores.tolist())

                    if args.removeMotionBranch == False:
                        scores_list = []
                        for clip_i in motion_feats_dict[video_name].keys():
                            error_clip = 0
                            for cls_id in motion_feats_dict[video_name][clip_i].keys():
                                feats = torch.from_numpy(motion_feats_dict[video_name][clip_i][cls_id]).cuda()
                                encoder_output, _ = model(feats, sign=1)
                                similarity = torch.matmul(encoder_output / args.temporature, motion_memories.T)
                                similarity_softmax = F.softmax(similarity, dim=1)
                                encoder_output = torch.matmul(similarity_softmax, motion_memories)
                                decoder_output = decoder(encoder_output, sign=1)
                                motion_feats = feats[:, :args.motion_feature_dim]
                                l2_error = calc_l2_error(decoder_output, motion_feats)
                                error_clip = max(error_clip, torch.max(l2_error).cpu().item())
                            motion_frames_scores[clip_i * args.segment_len:(clip_i + 1) * args.segment_len] = error_clip
                    motion_scores_list.extend(motion_frames_scores.tolist())
                    label_list.extend(gt[:n_clips * args.segment_len])

            appear_scores = min_max_norm(appear_scores_list)
            temp = np.concatenate((appear_scores[:padding_size], appear_scores, appear_scores[-padding_size:]))
            appear_scores = np.correlate(temp, filter_2d, 'valid')
            motion_scores = np.array(motion_scores_list)
            if args.removeMotionBranch == False:
                motion_scores = min_max_norm(motion_scores_list)
                temp = np.concatenate((motion_scores[:padding_size], motion_scores, motion_scores[-padding_size:]))
                motion_scores = np.correlate(temp, filter_2d, 'valid')

            frame_scores = appear_scores * 0.5 + motion_scores * 0.5
            frame_scores_list = frame_scores.tolist()

            auc_test = eval(frame_scores_list, label_list, None)
            print("test auc = ", auc_test)
            if auc_test > best_test_auc:
                best_test_auc = auc_test
                torch.save(model.state_dict(), os.path.join(args.model_saved_dir, args.dataset+"_encoder_epoch_" + str(epoch) + "_" + str(best_test_auc)))
                torch.save(decoder.state_dict(), os.path.join(args.model_saved_dir, args.dataset+"_decoder_epoch_" + str(epoch) + "_" + str(best_test_auc)))
                torch.save(MB.state_dict(), os.path.join(args.model_saved_dir, args.dataset+"_MB_epoch_" + str(epoch) + "_" + str(best_test_auc)))
            model = model.train()
            decoder = decoder.train()
        dataloader.dataset.shuffle_keys()

if __name__ == '__main__':
    args = parser_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seeds(args.seed)
    train(args)