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





imgvolume = Image.open('./DIC-C2DH-HeLa/trainrot.tif')
imglabel = Image.open('.//DIC-C2DH-HeLa/lableedgerot.tif')

imgindex = 0

trans = False
rot = True
flip = False
for i in range(16):
    try:
        imgvolume.seek(i)
        imglabel.seek(i)

        imgvolume.save('./DIC-C2DH-HeLa/trainrot/t%s.tif' % (imgindex,))
        imglabel.save('./DIC-C2DH-HeLa/labeledgerot/man_seg%s.tif' % (imgindex,))

        imgindex = imgindex + 1

        if rot:
            # use 10  steps ( 36 )
            for z in range(1, 36):
                angle = 360.0 / 36 * z

                rotate_img(imgvolume, angle, False).save('./DIC-C2DH-HeLa/trainrot/t%s.tif' % (imgindex,))
                rotate_img(imglabel, angle, True).save('./DIC-C2DH-HeLa/labeledgerot/man_seg%s.tif' % (imgindex,))

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

#
# img = Image.open('./test-volume.tif')
# directory = './ISBI 2012/Test-Volume/'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# for i in range(30):
#     try:
#         img.seek(i)
#         img.save('./ISBI 2012/Test-Volume/test-volume-%s.tif' % (i,))
#     except EOFError:
#         break
