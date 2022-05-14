import argparse
import os
import cv2 as cv
import csv
import random

display_classes_whitelist = [
    'display'
]

digits_classes_whitelist = [str(i) for i in range(10)]

orig_classes = [*display_classes_whitelist, *digits_classes_whitelist]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', '-d', type=str, required=True)
    parser.add_argument('--trainset_ratio', '-ts', type=float, required=True)
    parser.add_argument('--dataset_type', '-dt', type=str, required=True)
    return parser.parse_args()


def convert2display_boxes(boxes):
    tmp = list(filter(lambda x: orig_classes[int(x[0])] in display_classes_whitelist, boxes))
    for row in tmp:
        row[0] = str(display_classes_whitelist.index(orig_classes[int(row[0])]))
    if len(tmp) != 1:
        print("Something went wrong")
    return tmp


def yolo2cv(bbs, img):
    height, width = img.shape[:2]
    out_bbs = []
    for bb in bbs:
        tmp_bb = {}
        tmp_bb['label'] = bb[0]
        tmp_bb['lt'] = (round((float(bb[1]) - float(bb[3]) / 2) * width),
                        round((float(bb[2]) - float(bb[4]) / 2) * height))
        tmp_bb['rb'] = (round((float(bb[1]) + float(bb[3]) / 2) * width),
                        round((float(bb[2]) + float(bb[4]) / 2) * height))
        out_bbs.append(tmp_bb)
    return out_bbs

def yolo2cv_rel(bbs):
    out_bbs = []
    for bb in bbs:
        tmp_bb = {}
        tmp_bb['label'] = bb[0]
        tmp_bb['lt'] = ((float(bb[1]) - float(bb[3]) / 2),
                        (float(bb[2]) - float(bb[4]) / 2))
        tmp_bb['rb'] = ((float(bb[1]) + float(bb[3]) / 2),
                        (float(bb[2]) + float(bb[4]) / 2))
        out_bbs.append(tmp_bb)
    return out_bbs


def convert2digits_boxes(boxes, image):
    tmp = list(filter(lambda x: orig_classes[int(x[0])] in digits_classes_whitelist, boxes))
    for row in tmp:
        row[0] = str(digits_classes_whitelist.index(orig_classes[int(row[0])]))
    if len(tmp) == 0:
        return None
    bbs = yolo2cv(tmp, image)
    top = min(bbs, key=lambda x: x['lt'][1])['lt'][1]
    left = min(bbs, key=lambda x: x['lt'][0])['lt'][0]
    bottom = max(bbs, key=lambda x: x['rb'][1])['rb'][1]
    right = max(bbs, key=lambda x: x['rb'][0])['rb'][0]
    image = image[top:bottom,left:right]

    bbs_rel = yolo2cv_rel(tmp)

    top = min(bbs_rel, key=lambda x: x['lt'][1])['lt'][1]
    left = min(bbs_rel, key=lambda x: x['lt'][0])['lt'][0]
    bottom = max(bbs_rel, key=lambda x: x['rb'][1])['rb'][1]
    right = max(bbs_rel, key=lambda x: x['rb'][0])['rb'][0]

    ratio_x = 1 / (right - left)
    ratio_y = 1 / (bottom - top)

    for row in tmp:
        row[1] = str((float(row[1]) - left) * ratio_x)
        row[2] = str((float(row[2]) - top) * ratio_y)
        row[3] = str(float(row[3]) * ratio_x)
        row[4] = str(float(row[4]) * ratio_y)
    return image, tmp


if __name__ == '__main__':
    args = parse_args()
    images_path = os.path.join(args.dataset_root, 'obj_train_data')
    files = os.listdir(images_path)
    if args.dataset_type == 'display':
        files = list(filter(lambda x: x[-4:] == '.txt', files))
        for file in files:
            ann_path = os.path.join(images_path, file)
            with open(ann_path) as f:
                annotations = list(csv.reader(f, delimiter=' '))
            annotations = convert2display_boxes(annotations)
            with open(ann_path, 'w') as f:
                f.write('\n'.join([' '.join(box) for box in annotations]))
    elif args.dataset_type == 'digits':
        annotations = list(filter(lambda x: x[-4:] == '.txt', files))
        images = list(filter(lambda x: x[-4:] == '.jpg', files))
        if len(annotations) != len(images):
            raise RuntimeError("num of annotations and images must be the same")
        for annotation, image_name in zip(annotations, images):
            ann_path = os.path.join(images_path, annotation)
            img_path = os.path.join(images_path, image_name)
            with open(ann_path) as f:
                boxes = list(csv.reader(f, delimiter=' '))
            image = cv.imread(img_path, cv.IMREAD_COLOR)
            ret = convert2digits_boxes(boxes, image)
            if ret is None:
                os.remove(ann_path)
                os.remove(img_path)
                continue
            image, boxes = ret
            cv.imwrite(img_path, image)
            with open(ann_path, 'w') as f:
                f.write('\n'.join([' '.join(box) for box in boxes]))
    else:
        raise NameError('Dataset type must be "display" or "digits"')

    images_rel_path = 'data/obj_train_data'
    image_names = list(filter(lambda x: x.strip('.')[1] == 'jpg', files))

    random.shuffle(image_names)
    train_set_len = int(len(image_names) * args.trainset_ratio)
    train_set = image_names[:train_set_len]
    test_set = image_names[train_set_len:]

    with open(os.path.join(args.dataset_root, 'train.txt'), 'w') as file:
        file.write('\n'.join([os.path.join(images_rel_path, item) for item in train_set]))

    with open(os.path.join(args.dataset_root, 'valid.txt'), 'w') as file:
        file.write('\n'.join([os.path.join(images_rel_path, item) for item in test_set]))

    is_digits = args.dataset_type == 'digits'

    with open(os.path.join(args.dataset_root, 'obj.names'), 'w') as file:
        file.write('\n'.join(digits_classes_whitelist if is_digits else display_classes_whitelist))

    with open(os.path.join(args.dataset_root, 'obj.data'), 'w') as file:
        to_write = [
            f'classes = {10 if is_digits else 1}',
            f'train = data/train.txt',
            f'valid = data/valid.txt',
            f'names = data/obj.names',
            'backup = backup/'
        ]
        file.write('\n'.join(to_write))
