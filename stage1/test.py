# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images
from PIL import Image
def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        array = tensor.detach().numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        image = Image.fromarray(array)
        # image.show()
        image.save(os.path.join(save_dir, img_name + '.jpg'))

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="TryOn")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)

    parser.add_argument("--dataroot", default="")
    parser.add_argument("--datamode", default="testdata")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='results_test', help='save checkpoint infos')
   
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')


    opt = parser.parse_args()
    return opt


def train_pgp(opt, test_loader, model):
    model.cuda()
    model.eval()

    output_dir = os.path.join(opt.result_dir, opt.stage)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name_num = 0
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        print(step)
        save_name = inputs['pair']
        agnostic = inputs['agnostic'].cuda()  # torch.Size([19, 256, 192])
        target_parse = inputs['target_body_shape'].cuda()  # Image

        target_parse_visual = target_parse
        target_parse = target_parse.long()
        parse_shape = target_parse.shape
        batch_size = parse_shape[0]

        out_parse, softM_parse = model(agnostic)  # Image

        final_output = torch.argmax(softM_parse, dim=2)
        final_output = final_output.view(1, 256, 192)

        output_s = (final_output.squeeze().cpu()).numpy()

        output_s = abs(output_s) * 255
        output_save = Image.fromarray(output_s.astype('uint8'))
        output_save = output_save.convert('L')

        [save_path] = save_name
        name = 'target_softmax.jpg'
        output_save.save(os.path.join(output_dir, save_path + '.jpg'))
        # print(os.path.join(output_dir, save_path + '.jpg'))
        name_num += 1

        print("success save")


def train_gmm(opt, test_loader, model):
    model.cuda()
    model.train()

    # criterion
    model.cuda()
    model.eval()

    output_dir = os.path.join(opt.result_dir, opt.stage)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        c_names = inputs['pair']
        c_image = inputs['c_image'].cuda()
        t_pose = inputs['t_pose'].cuda()
        c_head = inputs['c_head'].cuda()
        target_head = inputs['tar_head'].cuda()
        Pre_target_mask = inputs['Pre_target_mask'].cuda()
        cloth = inputs['cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        agnostic = torch.cat([Pre_target_mask, t_pose, target_head], 1)

        grid, theta = model(agnostic, cloth)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')


        save_images(warped_cloth, c_names, output_dir)
        print(step)


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -max(0,step - opt.keep_step) / float(
        opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        outputs = model(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, cm * 2 - 1, m_composite * 2 - 1],
                   [p_rendered, p_tryon, im]]

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            board.add_scalar('L1', loss_l1.item(), step + 1)
            board.add_scalar('VGG', loss_vgg.item(), step + 1)
            board.add_scalar('MaskL1', loss_mask.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step + 1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))


from networks import PGP


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)
    print('//////////////////////')

    # visualization
    # if not os.path.exists(opt.tensorboard_dir):
    #     os.makedirs(opt.tensorboard_dir)
    # # board = SummaryWriter(log_dir = 'G:/work 2/Codes/cp-vton-master/tensorboard')


    # create model & train & save the final checkpoint
    if opt.stage == 'PGP':
        model = PGP(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_pgp(opt, train_loader, model)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'pgp_final.pth'))
    elif opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))

    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main()
