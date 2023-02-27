# coding:utf-8
# --------------------------------------------------------
# Visualization of data with VOC format
# Licensed under The MIT License [see LICENSE for details]
# Written by Yangguang Zhu
# --------------------------------------------------------
import os
import cv2
import re
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
pattens = ['name', 'xmin', 'ymin', 'xmax', 'ymax']


def get_annotations(xml_path):
    bbox = []
    with open(xml_path, 'r') as f:
        text = f.read().replace('\n', 'return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten, patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                elif patten == 'score':
                    tmp.append(float(p.findall(obj)[0]))
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox


def cv_text(im, bbox, class_name):
    """
    :param im:
    :param bbox: 
    :param class_name: 
    :return: 
    """
    pt1 = bbox[0:2]
    pt2 = bbox[2:4]
    text = '%s' % (class_name)
    fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
    fontScale = 1
    thickness = 1

    # plot the rectangle of bbox
    cv2.rectangle(im, bbox[0:2], bbox[2:4], thickness=2, color=(0, 255, 0))
    retval, baseLine = cv2.getTextSize(text, fontFace=fontFace, fontScale=fontScale, thickness=thickness)
    topleft = (pt1[0], pt1[1] - retval[1])
    bottomright = (topleft[0] + retval[0], topleft[1] + retval[1])

    cv2.rectangle(im, (topleft[0], topleft[1] - baseLine), bottomright, thickness=-1, color=(0, 255, 0))
    cv2.putText(im, text, (pt1[0], pt1[1]-baseLine),
                fontScale=fontScale, fontFace=fontFace, thickness=thickness, color=(0, 0, 0))

    return im



def save_viz_image(image_path, xml_path, save_path):
    bbox = get_annotations(xml_path)
    image_ = cv2.imread(image_path)
    image = np.copy(image_)
    for info in bbox:

        image = cv_text(image, tuple(info[1:5]), info[0])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite(os.path.join(save_path, image_path.split('/')[-1]), image)


if __name__ == '__main__':

    image_dir = '/workspace/dataset/HRRSD/JPEGImages/'
    xml_dir = '/workspace/dataset/HRRSD/Annotations/'
    save_dir = '/workspace/output_vis/hrrsd_gt'
    text_path = "/workspace/dataset/HRRSD/ImageSets/Main/test.txt"


    with open(text_path, 'r') as f:
        image_list = [(fl.strip('\n') + '.jpg') for fl in f.readlines()]
    f.close()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in tqdm(image_list, dynamic_ncols=True):
        image_path = os.path.join(image_dir, i)
        xml_path = os.path.join(xml_dir, i.replace('.jpg', '.xml'))
        save_viz_image(image_path, xml_path, save_dir)
