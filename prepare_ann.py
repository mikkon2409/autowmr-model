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


def convert2display_boxes(boxes, whitelist):
    tmp = list(filter(lambda x: orig_classes[int(x[0])] in whitelist, boxes))
    for row in tmp:
        row[0] = str(whitelist.index(orig_classes[int(row[0])]))
    if len(tmp) != 1:
        print("Something went wrong")
    return tmp


if __name__ == '__main__':
    args = parse_args()

    if args.dataset_type == 'display':
        images_path = os.path.join(args.dataset_root, 'obj_train_data')
        files = os.listdir(images_path)
        files = list(filter(lambda x: x[-4:] == '.txt', files))
        for file in files:
            ann_path = os.path.join(images_path, file)
            with open(ann_path) as f:
                annotations = list(csv.reader(f, delimiter=' '))
            annotations = convert2display_boxes(annotations, display_classes_whitelist)
            with open(ann_path + '.new', 'w') as f:
                f.write('\n'.join([' '.join(box) for box in annotations]))
    elif args.dataset_type == 'digits':
        pass
    else:
        raise NameError('Dataset type must be "display" or "digits"')


    with open(os.path.join(args.dataset_root, 'train.txt')) as f:
        entities = f.readlines()
    random.shuffle(entities)
    train_set_len = int(len(entities) * args.trainset_ratio)
    train_set = entities[:train_set_len]
    test_set = entities[train_set_len:]

    with open(os.path.join(args.dataset_root, 'train.txt.new'), 'w') as file:
        file.write(''.join(train_set))

    with open(os.path.join(args.dataset_root, 'valid.txt.new'), 'w') as file:
        file.write(''.join(test_set))

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
