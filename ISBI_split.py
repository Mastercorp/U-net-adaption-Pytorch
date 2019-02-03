from PIL import Image
import os
from skimage.transform import AffineTransform, warp, rotate
import numpy as np

# For the Dataset register at : http://brainiac2.mit.edu/isbi_challenge/
# Download the corresponding data files
# Only 30 images are available with ground truth
# 6 Images are used for validation and are put into a seperate folder


def affinetrans(img, x, y):
    tform = AffineTransform(translation=(x, y))
    imgaug = np.asarray(img)
    imgaug = Image.fromarray((warp(imgaug, tform, mode='reflect') * 255).astype(np.uint8))
    return imgaug

# rotate and do not allow values between 0 and 255
def rotate_img(img, angle, label):
    imgaug = np.asarray(img)
    imgaug = rotate(imgaug, angle, mode='reflect')
    if label:
        low_values_flag = imgaug <= 0.5
        high_values_flag = imgaug > 0.5
        imgaug[low_values_flag] = 0
        imgaug[high_values_flag] = 1
    imgaug = (imgaug*255).astype(np.uint8)
    return Image.fromarray(imgaug)


directory = './ISBI 2012/Train-Volume/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = './ISBI 2012/Val-Volume/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = './ISBI 2012/Train-Labels/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory = './ISBI 2012/Val-Labels/'
if not os.path.exists(directory):
    os.makedirs(directory)


imgvolume = Image.open('./train-volume.tif')
imglabel = Image.open('./train-labels.tif')

imgindex = 0

trans = False
rot = False
flip = False
for i in range(30):
    try:
        imgvolume.seek(i)
        imglabel.seek(i)

        if i % 3 == 0:
            imgvolume.save('./ISBI 2012/Val-Volume/train-volume-%s.tif' % (imgindex,))
            imglabel.save('./ISBI 2012/Val-Labels/train-labels-%s.tif' % (imgindex,))

        else:
            imgvolume.save('./ISBI 2012/Train-Volume/train-volume-%s.tif' % (imgindex,))
            imglabel.save('./ISBI 2012/Train-Labels/train-labels-%s.tif' % (imgindex,))

        imgindex = imgindex + 1

        if rot:
            # use 10  steps ( 36 )
            for z in range(1, 36):
                angle = 360.0 / 36 * z
                if i % 3 == 0:
                    continue
                    # rotate_img(imgvolume, angle, False).save('./ISBI 2012/Val-Volume/train-volume-%s.tif' % (imgindex,))
                    # rotate_img(imglabel, angle, True).save('./ISBI 2012/Val-Labels/train-labels-%s.tif' % (imgindex,))


                else:
                    rotate_img(imgvolume, angle, False).save('./ISBI 2012/Train-Volume/train-volume-%s.tif' % (imgindex,))
                    rotate_img(imglabel, angle, True).save('./ISBI 2012/Train-Labels/train-labels-%s.tif' % (imgindex,))

                imgindex = imgindex + 1

        if flip:
            for k in range(1, 4):
                angle = 90 * k
                if i % 5 == 0:
                    rotate_img(imgvolume, angle, False).save('./ISBI 2012/Val-Volume/train-volume-%s.tif' % (imgindex,))
                    rotate_img(imglabel, angle, True).save('./ISBI 2012/Val-Labels/train-labels-%s.tif' % (imgindex,))
                else:
                    rotate_img(imgvolume, angle, False).save('./ISBI 2012/Train-Volume/train-volume-%s.tif' % (imgindex,))
                    rotate_img(imglabel, angle, True).save('./ISBI 2012/Train-Labels/train-labels-%s.tif' % (imgindex,))
                imgindex = imgindex + 1

        if trans:
            x = 3
            y = 3
            for x1 in range(-x, x + 1, 2):
                for y1 in range(-y, y + 1, 2):
                    if i % 5 == 0:
                        affinetrans(imgvolume, x1, y1).save('./ISBI 2012/Val-Volume/train-volume-%s.tif' % (imgindex,))
                        affinetrans(imglabel, x1, y1).save('./ISBI 2012/Val-Labels/train-labels-%s.tif' % (imgindex,))
                    else:
                        affinetrans(imgvolume, x1, y1).save('./ISBI 2012/Train-Volume/train-volume-%s.tif' % (imgindex,))
                        affinetrans(imglabel, x1, y1).save('./ISBI 2012/Train-Labels/train-labels-%s.tif' % (imgindex,))
                    imgindex = imgindex + 1
    except EOFError:
        break


img = Image.open('./test-volume.tif')
directory = './ISBI 2012/Test-Volume/'
if not os.path.exists(directory):
    os.makedirs(directory)
for i in range(30):
    try:
        img.seek(i)
        img.save('./ISBI 2012/Test-Volume/test-volume-%s.tif' % (i,))
    except EOFError:
        break
