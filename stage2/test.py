# coding=utf-8

import argparse
import os
from PIL import Image
from tensorboardX import SummaryWriter
import time
from networks import save_checkpoint, VGGLoss
from cp_dataset import CPDataset, CPDataLoader
from TryOn import cyclegan
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Try-On")
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)

    # parser.add_argument("--dataroot", default="/media/a/新加卷/work1/Codes")
    # parser.add_argument("--datamode", default="testdata")
    parser.add_argument("--dataroot", default="")
    parser.add_argument("--datamode", default="testdata")
    parser.add_argument("--data_list", default="test_pairs.txt")
    parser.add_argument("--stage", default="Try-On")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='save checkpoint infos')
    parser.add_argument('--test', type=str, default='test', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--gan_mode", type=str, default='wgangp')
    opt = parser.parse_args()
    return opt


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()
import itertools
from utils import *
def test_tryon(opt, train_loader, model):

    model.cuda()
    model.eval()


    for step, inputs in enumerate(train_loader.data_loader):
        iter_start_time = time.time()

        pairs = inputs["pair"]
        print(step, '++++++')
        print("-------------------------------------")
        model.set_input(inputs)
        model.optimize_parameters()
        results = model.current_results()

        print('step: %8d, G_loss: %4f, c_loss: %4f' % (
            step + 1, results['G_loss'].item(),  results['content_loss'].item()), flush=True)


        isExists = os.path.exists(os.path.join(opt.test, opt.name))
        if not isExists:
            os.makedirs(os.path.join(opt.test, opt.name))
        save_images(results['gen_B'], pairs, os.path.join(opt.test, opt.name))

        # # print(name1)

def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        # array = tensor.numpy().astype('uint8')
        array = tensor.detach().numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        image = Image.fromarray(array)
        # image.show()
        image.save(os.path.join(save_dir, img_name + '.jpg'))

def main():
    opt = get_opt()
    print(opt)
    print("named: %s!" % (opt.name))

    # create dataset
    train_dataset = CPDataset(opt)
    generator = cyclegan(opt)
    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(generator, opt.checkpoint)
    test_tryon(opt, train_loader, generator)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main()



def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

