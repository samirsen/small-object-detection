import os
import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage


from utils.config import Config
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at

from SRGAN import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=1, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

FasterRCNNOpt = Config()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
MODEL_NAME = opt.model_name

gan_model = Generator(UPSCALE_FACTOR).eval()
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn)


if TEST_MODE:
    gan_model.cuda()
    trainer.cuda()
    gan_model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    trainer.load('epochs/samir_fast_rcnn_epoch60.pth')

else:
    gan_model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    trainer.load('epochs/samir_fast_rcnn_epoch60.pth', map_location=lambda storage, loc: storage))

image = read_image('misc/demo.jpg')
image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

start = time.clock()
out = gan_model(image)
out_img = ToPILImage()(out[0].data.cpu())
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)

_bboxes, _labels, _scores = trainer.faster_rcnn.predict(out_img, visualize=True)
ax = vis_bbox(at.tonumpy(img[0]),
         at.tonumpy(_bboxes[0]),
         at.tonumpy(_labels[0]).reshape(-1),
         at.tonumpy(_scores[0]).reshape(-1))

plt.show()

elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
# out_img = ToPILImage()(out[0].data.cpu())
# out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
