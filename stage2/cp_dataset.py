# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json
import pickle


class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """

    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.stage = opt.stage  # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # load data list
        pairs = []
        c_names = []
        t_names = []

        with open(opt.data_list, 'r') as f:
            for line in f.readlines():
                pair, c_name, t_name = line.strip().split()
                pairs.append(pair)
                c_names.append(c_name)
                t_names.append(t_name)

        self.pairs = pairs
        self.c_names = c_names
        self.t_names = t_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        pair = self.pairs[index]  # the index of the pair
        c_name = self.c_names[index]  # condition person index
        t_name = self.t_names[index]  # target person index

        # cloth image & cloth mask

        path_cloth = osp.join(self.data_path, pair, "target.jpg")
        path_mask = osp.join(self.data_path, pair, "mask.jpg")
        cloth = Image.open(path_cloth)
        cloth_mask = Image.open(path_mask)

        target_softmax_path = osp.join("results_test/PGP", pair+ '.jpg')
        target_softmax_shape = Image.open(target_softmax_path)
        target_softmax_shape = target_softmax_shape.resize((self.fine_width // 4, self.fine_height // 4), Image.BILINEAR)
        target_softmax_shape = target_softmax_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        target_softmax_shape = self.transform(target_softmax_shape)

        path_c_image = osp.join(self.data_path, pair, c_name + ".jpg")  # condition image path
        path_t_image = osp.join(self.data_path, pair, t_name + "_target.jpg")  # target image path
        path_warp_cloth = osp.join("results_test/GMM", pair+ '.jpg')
        c_image = Image.open(path_c_image)
        t_image = Image.open(path_t_image)
        warp_cloth = Image.open(path_warp_cloth)
        c_mask_array = np.array(cloth_mask)
        c_mask_array = (c_mask_array > 0).astype(np.float32)
        c_mask = torch.from_numpy(c_mask_array)
        cloth_mask = c_mask.unsqueeze_(0)

        cloth = self.transform(cloth)
        c_image = self.transform(c_image)
        t_image = self.transform(t_image)
        warp_cloth = self.transform(warp_cloth)
        # conditionnal parsing and pose path
        path_c_seg = osp.join(self.data_path, pair, c_name, "segment.png")
        path_c_pose = osp.join(self.data_path, pair, c_name, "pose.pkl")

        # target parsing and pose path
        t_name = t_name + "_target"
        path_t_seg = osp.join(self.data_path, pair, t_name, "segment.png")
        path_t_pose = osp.join(self.data_path, pair, t_name, "pose.pkl")

        # conditional segment processing
        c_seg = Image.open(path_c_seg)
        c_parse_array = np.array(c_seg)
        c_parse_fla = c_parse_array.reshape(49152, 1)
        c_parse_fla = torch.from_numpy(c_parse_fla).long()
        c_parse = torch.zeros(49152, 20).scatter_(1, c_parse_fla, 1)
        c_parse = c_parse.view(256, 192, 20)
        c_parse = c_parse.transpose(2, 0).transpose(2, 1).contiguous()  # [20,256,192]

        c_shape = (c_parse_array > 0).astype(np.float32)  # condition body shape
        c_head = (c_parse_array == 1).astype(np.float32) + \
                 (c_parse_array == 2).astype(np.float32) + \
                 (c_parse_array == 4).astype(np.float32) + \
                 (c_parse_array == 13).astype(np.float32)
        c_cloth = (c_parse_array == 5).astype(np.float32) + \
                  (c_parse_array == 6).astype(np.float32) + \
                  (c_parse_array == 7).astype(np.float32)

        c_body_shape = c_shape
        c_shape_array = np.asarray(c_body_shape)
        c_body_fla = c_shape_array.reshape(49152, 1)
        c_body_fla = torch.from_numpy(c_body_fla).long()
        one_hot_c = torch.zeros(49152, 2).scatter_(1, c_body_fla, 1)
        one_hot_c = one_hot_c.view(256, 192, 2)
        one_hot_c = one_hot_c.transpose(2, 0).transpose(1, 2).contiguous()

        c_shape = Image.fromarray((c_shape * 255).astype(np.uint8))
        c_shape_sample = c_shape.resize((self.fine_width // 4, self.fine_height // 4), Image.BILINEAR)
        c_shape_sample = c_shape_sample.resize((self.fine_width, self.fine_height), Image.BILINEAR)

        c_cloth_sample = Image.fromarray((c_cloth * 255).astype(np.uint8))  # downsampling of cloth on the person
        c_cloth_sample = c_cloth_sample.resize((self.fine_width // 4, self.fine_height // 4), Image.BILINEAR)
        c_cloth_sample = c_cloth_sample.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        c_shape_sample = self.transform(c_shape_sample)

        c_head = torch.from_numpy(c_head)
        c_cloth = torch.from_numpy(c_cloth)
        c_cloth_sample = self.transform(c_cloth_sample)
        # c_cloth_sample = self.transform(c_cloth_sample)

        t_seg = Image.open(path_t_seg)
        t_parse_array = np.array(t_seg)
        t_parse_fla = t_parse_array.reshape(49152, 1)
        t_parse_fla = torch.from_numpy(t_parse_fla).long()
        t_parse = torch.zeros(49152, 20).scatter_(1, t_parse_fla, 1)
        t_parse = t_parse.view(256, 192, 20)
        t_parse = t_parse.transpose(2, 0).transpose(1, 2).contiguous()

        t_shape_mask = (t_parse_array > 0).astype(np.float32)
        t_head_mask = (t_parse_array == 1).astype(np.float32) + \
                      (t_parse_array == 2).astype(np.float32) + \
                      (t_parse_array == 4).astype(np.float32) + \
                      (t_parse_array == 13).astype(np.float32)
        t_cloth_mask = (t_parse_array == 5).astype(np.float32) + \
                       (t_parse_array == 6).astype(np.float32) + \
                       (t_parse_array == 7).astype(np.float32)

        # to obtain the one hot of t_shape, the value of each pixel is 1 or 0
        t_body_parse = t_shape_mask
        t_shape_array = np.asarray(t_body_parse)
        t_body_fla = t_shape_array.reshape(49152, 1)
        t_body_fla = torch.from_numpy(t_body_fla).long()
        one_hot_t = torch.zeros(49152, 2).scatter_(1, t_body_fla, 1)
        one_hot_t = one_hot_t.view(256, 192, 2)
        one_hot_t = one_hot_t.transpose(2, 0).transpose(1, 2).contiguous()  # [2,256,192]

        t_shape_mask = Image.fromarray((t_shape_mask * 255).astype(np.uint8))
        t_shape = self.transform(t_shape_mask)
        t_shape_mask = t_shape_mask.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
        t_shape_mask = t_shape_mask.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        t_shape_mask = self.transform(t_shape_mask)
        t_head_mask = torch.from_numpy(t_head_mask)
        t_cloth = torch.from_numpy(t_cloth_mask)
        con_head = c_image * c_head - (1 - c_head)
        con_cloth = c_image * c_cloth + (1 - c_cloth)

        tar_head = t_image * t_head_mask - (1 - t_head_mask)
        tar_cloth = t_image * t_cloth + (1 - t_cloth)

        c_pose_data = - np.ones((18, 2), dtype=int)
        with open(path_c_pose, 'rb') as f:
            pose_label = pickle.load(f)
            for i in range(18):
                if pose_label['subset'][0, i] != -1:
                    c_pose_data[i, :] = pose_label['candidate'][int(pose_label['subset'][0, i]), :2]
            c_pose_data = np.asarray(c_pose_data)

        point_num = 18
        c_pose_map = torch.zeros(point_num, 256, 192)
        r = self.radius
        c_pose = Image.new('L', (192, 256))
        c_pose_draw = ImageDraw.Draw(c_pose)
        for i in range(point_num):
            one_map = Image.new('L', (192, 256))
            draw = ImageDraw.Draw(one_map)
            c_pointx = c_pose_data[i, 0]

            c_pointy = c_pose_data[i, 1]
            c_pointx = c_pointx * 192 / 762
            c_pointy = c_pointy * 256 / 1000
            if c_pointx > 1 and c_pointy > 1:
                draw.rectangle((c_pointx - r, c_pointy - r, c_pointx + r, c_pointy + r), 'white', 'white')
                c_pose_draw.rectangle((c_pointx - r, c_pointy - r, c_pointx + r, c_pointy + r), 'white', 'white')
            one_map = self.transform(one_map)
            c_pose_map[i] = one_map[0]

        t_pose_data = - np.ones((18, 2), dtype=int)
        with open(path_t_pose, 'rb') as f:
            pose_label = pickle.load(f)
            for i in range(18):
                if pose_label['subset'][0, i] != -1:
                    t_pose_data[i, :] = pose_label['candidate'][int(pose_label['subset'][0, i]), :2]
            t_pose_data = np.asarray(t_pose_data)

        point_num = 18
        t_pose_map = torch.zeros(point_num, 256, 192)
        r = self.radius
        t_pose = Image.new('L', (192, 256))
        t_pose_1 = Image.new('L', (192, 256))
        t_pose_draw = ImageDraw.Draw(t_pose)
        t_pose_draw_1 = ImageDraw.Draw(t_pose_1)
        for i in range(point_num):
            one_map = Image.new('L', (192, 256))
            draw = ImageDraw.Draw(one_map)
            t_pointx = t_pose_data[i, 0]

            t_pointy = t_pose_data[i, 1]
            t_pointx = t_pointx * 192 / 762
            t_pointy = t_pointy * 256 / 1000
            if t_pointx > 1 and t_pointy > 1:
                draw.rectangle((t_pointx - r, t_pointy - r, t_pointx + r, t_pointy + r), 'white', 'white')
                t_pose_draw.rectangle((t_pointx - r, t_pointy - r, t_pointx + r, t_pointy + r), 'white', 'white')
                if i < 16:
                    t_pose_draw_1.rectangle((t_pointx - r, t_pointy - r, t_pointx + r, t_pointy + r), 'white', 'white')
            one_map = self.transform(one_map)
            t_pose_map[i] = one_map[0]

        # just for visualization
        # t_pose.show()
        # t_pose_1.show()
        c_v_pose = self.transform(c_pose)
        t_v_pose = self.transform(t_pose)

        # cloth-agnostic representation
        c_shape = self.transform(c_shape)
        agnostic = torch.cat([c_shape, t_pose_map], 0)
        agnostic_sample = torch.cat([c_shape_sample, t_pose_map], 0)

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''
        c_cloth = c_cloth.view(1, 256, 192)

        result = {
            'pair': pair,
            'cloth': cloth,
            'cloth_mask': cloth_mask,
            'c_image': c_image,  # conditional image
            't_image': t_image,
            'c_upper_cloth': con_cloth,  #
            't_upper_cloth': tar_cloth,
            'c_head': con_head,
            'c_pose': c_pose_map,
            'c_pose_v': c_v_pose,  # for visualization
            't_pose': t_pose_map,
            't_pose_v': t_v_pose,  # for visualization
            'c_shape': c_shape,
            't_shape': t_shape,
            'c_shape_sample': c_shape_sample,
            't_shape_sample': t_shape_mask,
            'agnostic': agnostic,
            'agnostic_sample': agnostic_sample,
            'grid_image': im_g,
            'Pre_t_shape': target_softmax_shape,
            'one_hot_c': one_hot_c,
            'one_hot_t': one_hot_t,
            't_upper_mask': t_cloth,
            't_head_mask': t_head_mask,
            'c_upper_mask': c_cloth,
            'c_head_mask': c_head,
            'target_body_shape': t_body_parse,
            'warp_cloth': warp_cloth,
        }
        # print(t_shape.shape, c_shape.shape, t_shape_mask.shape, im_g.shape)
        return result

    def __len__(self):
        return len(self.pairs)


class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)

    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
          % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed;

    embed()

