import glob

import cv2
import dicom
import numpy as np


def normalize(im, flip=False):
    # returns image with normalized values in [0,1]
    # flip switch inverts image (default False)
    im = (im - im.min()) / (im.max() - im.min())
    if flip:
        im = 1 - im
    return im


def get_box(im):
    im = normalize(im) * 255
    if len(im.shape) > 2:
        ret, thresh = cv2.threshold(im[:,:,0].astype(np.uint8), 1, 255, 0)
    else:
        ret, thresh = cv2.threshold(im.astype(np.uint8), 1, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnt)

    sub = im[y:y + h, x:x + w]
    return sub


def im_from_row(row, n=0, data_path='E:/Blinder/'):
    # returns image from n-th exam file
    file_loc = row.STUDY_LOCATION.replace('\\', '/')
    files = glob.glob(data_path + file_loc + '/*.*')
    file_1 = dicom.read_file(files[n])
    im = file_1.pixel_array
    # im_lowres = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(im)))
    return im


def dicom_from_row(row, n=0, data_path='E:/Blinder/'):
    # returns dicom metadata from n-th exam file corresponding to dataframe row
    file_loc = row.STUDY_LOCATION.replace('\\', '/')
    files = glob.glob(data_path + file_loc + '/*.*')
    dicom_file = dicom.read_file(files[n])
    return dicom_file


def box_from_row(row, n=0, data_path='E:/Blinder/'):
    dic = dicom_from_row(row, n, data_path)
    im = im_from_row(row, n, data_path)
    flip = False
    if dic.PhotometricInterpretation == 'MONOCHROME1':
        flip = True
    im = normalize(im, flip=flip)
    box = normalize(get_box(im))
    return (box)


def square(im):
    y, x = im.shape[0], im.shape[1]
    if (x > y):
        pad = int(np.ceil(0.5 * (x - y)))
        top, bottom, left, right = pad, pad, 0, 0
    elif (x < y):
        pad = int(np.ceil(0.5 * (y - x)))
        top, bottom, left, right = 0, 0, pad, pad
    else:
        top, bottom, left, right = 0, 0, 0, 0
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return (new_im)


def resize(im, new_x=224, new_y=224):
    return cv2.resize(im, (new_x, new_y))


def vert_flip(im):
    vert_flip_im = cv2.flip(im, 1)
    return vert_flip_im


def gray2rgb(im):
    rgb_im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    return rgb_im
