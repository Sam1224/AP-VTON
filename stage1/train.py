#coding=utf-8
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


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "")
    parser.add_argument("--datamode", default = "traindata")
    parser.add_argument("--stage", default = "PGP")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def train_pgp(opt, train_loader, model, board):
    model.cuda()
    model.train()


    supervision_softmax = nn.CrossEntropyLoss()
    L1 = nn.L1Loss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()

        inputs = train_loader.next_batch()

        agnostic = inputs['agnostic'].cuda()  # torch.Size([19, 256, 192])
        target_parse = inputs['target_body_shape'].cuda()  # Image
        t_parse = target_parse
        t_parse = t_parse.view(-1,1,256,192)
        target_parse_visual = target_parse
        target_parse = target_parse.long()
        parse_shape = target_parse.shape
        batch_size = parse_shape[0]

        target_parse = target_parse.view(batch_size, 256*192)
        out_parse, softM_parse = model(agnostic)  # Image
        final_output = torch.argmax(softM_parse, dim=2)

        final_output = final_output.view(batch_size, 1, 256, 192)
        f_result = final_output.float()
        out_parse = out_parse.view(-1, 2)
        target_parse = target_parse.view(-1)
        loss_soft = torch.mean(supervision_softmax(out_parse,target_parse))
        l1 = L1(f_result, t_parse)
        loss = loss_soft + l1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        target_parse_visual = target_parse_visual.view(batch_size, 1, 256, 192)
        visuals = [[final_output, target_parse_visual],]
        # loss = loss / (256 * 192)
        print(loss, l1, loss_soft)
        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss.item()), flush=True)
        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.stage, 'step_%06d.pth' % (step+1)))

def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        c_image = inputs['c_image'].cuda()
        t_pose = inputs['t_pose'].cuda()
        c_head = inputs['c_head'].cuda()
        Pre_target_mask = inputs['Pre_target_mask'].cuda()
        cloth = inputs['cloth'].cuda()
        cloth_mask = inputs['cloth_mask'].cuda()
        t_upper_mask = inputs['t_upper_mask'].cuda()
        t_upper_cloth = inputs['t_upper_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
        # agnostic = inputs['agnostic'].cuda()
        agnostic = torch.cat([Pre_target_mask,t_pose],1)
        grid, theta = model(agnostic, cloth)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
        warped_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
        visuals = [[c_head, Pre_target_mask, t_pose],
                   [cloth_mask, warped_cloth, t_upper_mask],
                   [warped_grid, (warped_cloth+c_image)*0.5, c_image]]

        loss = criterionL1(warped_cloth, t_upper_cloth)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:

            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))
    
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
        
        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose], 
                   [c, cm*2-1, m_composite*2-1], 
                   [p_rendered, p_tryon, im]]
            
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f' 
                    % (step+1, t, loss.item(), loss_l1.item(), 
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

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
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'PGP':
        model = PGP(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_pgp(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'pgp_final.pth'))
    elif opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))

    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
        
  
    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main()
