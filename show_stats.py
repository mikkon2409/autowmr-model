from __future__ import annotations
from pathlib import Path
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def read_annotations_path(filename, dataset_root):
    annotations_path = pd.read_csv(filename, header=None, names=['filepath'])
    annotations_path['filepath'] = annotations_path['filepath'].map(lambda x: Path(
        dataset_root, 'obj_train_data', Path(x).with_suffix('.txt').name))
    return annotations_path


def get_stats(annotations_path, class_names):
    stat = pd.DataFrame()
    for ann_file in annotations_path['filepath']:
        stat = pd.concat([stat, pd.read_csv(ann_file, sep=' ', usecols=[0], names=['label'], header=None, dtype=str)], axis=0)
    stat['label'] = stat['label'].map(lambda x: class_names[int(x)])

    value_counts = stat['label'].value_counts()
    names = value_counts.index.array
    values = value_counts.array
    return names, values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', '-d', type=str, required=True)
    return parser.parse_args()


args = parse_args()

dataset_path = Path(args.dataset_root)
if not dataset_path.exists():
    raise FileExistsError("'dataset_root' doesn't exists")
train_path = Path(dataset_path, 'train.txt')
if not train_path.exists():
    raise FileExistsError("'train.txt' doesn't exists")
valid_path = Path(dataset_path, 'valid.txt')
if not train_path.exists():
    raise FileExistsError("'valid.txt' doesn't exists")
names_path = Path(dataset_path, 'obj.names')
if not names_path.exists():
    raise FileExistsError("'obj.names' doesn't exists")

names = pd.read_csv(names_path, header=None, names=['classes'])['classes']

train_annotations_path = read_annotations_path(train_path, dataset_path)
print(train_annotations_path)

valid_annotations_path = read_annotations_path(valid_path, dataset_path)
print(valid_annotations_path)

train_names, train_values = get_stats(train_annotations_path, names)
valid_names, valid_values = get_stats(valid_annotations_path, names)

colors = sns.color_palette('pastel')[0:len(names)]

fig = plt.figure(1, figsize=(15, 15))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.pie(train_values, labels = train_names, colors = colors, autopct='%.2f%%')
ax1.title.set_text('Train set')
ax2.pie(valid_values, labels = valid_names, colors = colors, autopct='%.2f%%')
ax2.title.set_text('Validation set')
plt.show()
