import glob
from torch.utils import data
from PIL import Image
import torchvision
import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import datetime

class ISBIDataset(data.Dataset):

    def __init__(self, gloob_dir_train, gloob_dir_label, length, is_pad, evaluate, totensor):
        self.gloob_dir_train = gloob_dir_train
        self.gloob_dir_label = gloob_dir_label
        self.length = length
        self.crop = torchvision.transforms.CenterCrop(512)
        self.crop_nopad = torchvision.transforms.CenterCrop(324)
        self.is_pad = is_pad
        self.evaluate = evaluate
        self.totensor = totensor
        self.changetotensor = torchvision.transforms.ToTensor()

        self.trainfiles = sorted(glob.glob(self.gloob_dir_train),
                            key=lambda name: int(name[self.gloob_dir_train.rfind('*'):
                                                      -(len(self.gloob_dir_train) - self.gloob_dir_train.rfind('.'))]))

        self.labelfiles = sorted(glob.glob(self.gloob_dir_label),
                            key=lambda name: int(name[self.gloob_dir_label.rfind('*'):
                                                      -(len(self.gloob_dir_label) - self.gloob_dir_label.rfind('.'))]))

        # self.rand_vflip = False
        # self.rand_hflip = False
        # self.rand_rotate = False
        # self.angle = 0

    def __len__(self):
        'Denotes the total number of samples'
        return self.length

    def __getitem__(self, index):
        'Generates one sample of data'
        # files are sorted depending the last number in their filename
        # for example : "./ISBI 2012/Train-Volume/train-volume-*.tif"
        # np.random is not thread safe, therefore creating the same random numbers every epoche
        # add random seed to each item
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        trainimg = Image.open(self.trainfiles[index]).convert("L")
        trainlabel = Image.open(self.labelfiles[index]).convert("L")


        if not self.evaluate:

            sigma = random.randint(70, 100)


            if random.random() < 0.5:
                trainlabel = trainlabel.transpose(Image.FLIP_LEFT_RIGHT)
                trainimg = trainimg.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() < 0.5:
                trainlabel = trainlabel.transpose(Image.FLIP_TOP_BOTTOM)
                trainimg = trainimg.transpose(Image.FLIP_TOP_BOTTOM)

            if random.random() < 0.5:
                timg = np.asarray(trainimg)
                limg = np.asarray(trainlabel)

                dx = np.gradient(timg.astype('float32'), axis=0)
                dy = np.gradient(timg.astype('float32'), axis=1)

                # maybe error, why axis=0?
                # dx /= np.max(np.abs(dx), axis=0)
                # dy /= np.max(np.abs(dy), axis=0)

                # divide by max abs value of whole matrix
                dx /= np.max(np.abs(dx))
                dy /= np.max(np.abs(dy))

                # to shift in gradient direction -, else it shifts in the other direction
                scalex = random.randint(1, 10) * 1000
                scaley = random.randint(1, 10) * 1000
                if random.random() < 0.5:
                    scalex = scalex * -1
                if random.random() < 0.5:
                    scaley = scaley * -1

                dxconv = gaussian_filter(dx, sigma, order=0, mode='mirror', truncate=3) * -scalex
                dyconv = gaussian_filter(dy, sigma, order=0, mode='mirror', truncate=3) * -scaley

                x, y = np.meshgrid(np.arange(timg.shape[1]), np.arange(timg.shape[0]), indexing='xy')
              #  row_norm = np.linalg.norm([np.reshape(dxconv, (-1, 1)), np.reshape(dyconv, (-1, 1))], axis=0)

                indices = [np.reshape(y + dyconv, (-1, 1)), np.reshape(x + dxconv, (-1, 1))]

                trainimg = Image.fromarray(
                    map_coordinates(timg, indices, order=1, mode='mirror').reshape((timg.shape[0], timg.shape[1])))

                trainlabel = map_coordinates(limg, indices, order=1, mode='mirror').reshape((limg.shape[0], limg.shape[1]))

                low_values_flag = trainlabel <= 127
                high_values_flag = trainlabel > 127
                trainlabel[low_values_flag] = 0
                trainlabel[high_values_flag] = 1
                trainlabel = Image.fromarray((trainlabel * 255).astype(np.uint8))


            # if self.rand_rotate:
            #     # Add padding to the image to remove black boarders when rotating
            #     # image is croped to true size later.
            #     trainimg = Image.fromarray(np.pad(np.asarray(trainimg), ((107, 107), (107, 107)), 'reflect'))
            #     trainlabel = Image.fromarray(np.pad(np.asarray(trainlabel), ((107, 107), (107, 107)), 'reflect'))
            #
            #     trainlabel = trainlabel.rotate(self.angle)
            #     trainimg = trainimg.rotate(self.angle)
            #     # crop rotated image to true size
            #     trainlabel = self.crop(trainlabel)
            #     trainimg = self.crop(trainimg)




        # when padding is used, dont crop the label image
        if not self.is_pad:
            trainlabel = self.crop_nopad(trainlabel)

        if self.totensor:
            # test if NLL needs long
            trainlabel = self.changetotensor(trainlabel).long()
            trainimg = self.changetotensor(trainimg)

        return trainimg, trainlabel
