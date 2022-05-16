import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import cv2 as cv

classes_whitelist = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', '-d', type=str, required=True)
    parser.add_argument('--trainset_percentage', '-tp',
                        type=float, required=True)
    parser.add_argument('--random_state', '-rs',
                        type=int, required=False, default=None)
    return parser.parse_args()


def clip(value, min_thresh, max_thresh):
    return min(max_thresh, max(min_thresh, value))


args = parse_args()
trainset_percentage = args.trainset_percentage
if trainset_percentage < 0 or trainset_percentage > 1:
    raise RuntimeError("'trainset_percentage' must be in [0.0; 1.0] range")

dataset_path = Path(args.dataset_root)
if not dataset_path.exists():
    raise FileExistsError("'dataset_root' doesn't exists")
obj_train_data_path = Path(dataset_path, 'obj_train_data')
if not obj_train_data_path.exists():
    raise FileExistsError("'obj_train_data' doesn't exists")
train_path = Path(dataset_path, 'train.txt')
if not train_path.exists():
    raise FileExistsError("'train.txt' doesn't exists")
obj_data_path = Path(dataset_path, 'obj.data')
if not obj_data_path.exists():
    raise FileExistsError("'obj.data' doesn't exists")
obj_names_path = Path(dataset_path, 'obj.names')
if not obj_names_path.exists():
    raise FileExistsError("'obj.names' doesn't exists")

valid_path = Path(dataset_path, 'valid.txt')

class_names = pd.read_csv(obj_names_path, header=None, names=[
                          'class_name'], dtype=str)['class_name']


annotation_files_paths = obj_train_data_path.glob('*.txt')
for annotation_file in tqdm(annotation_files_paths):
    annotation = pd.read_csv(annotation_file, sep=' ', header=None, names=[
                             'label', 'cx', 'cy', 'w', 'h'])
    annotation['label'] = annotation['label'].map(class_names)
    annotation = annotation[annotation['label'].isin(classes_whitelist)]
    annotation['label'] = annotation['label'].map(
        lambda x: classes_whitelist.index(x))
    if len(annotation) < 1:
        annotation_file.unlink()
        annotation_file.with_suffix('.jpg').unlink()
        print(f'Removed {annotation_file}')
        continue
    annotation = pd.concat([annotation['label'],
                            annotation['cx'] - annotation['w'] / 2,
                            annotation['cy'] - annotation['h'] / 2,
                            annotation['cx'] + annotation['w'] / 2,
                            annotation['cy'] + annotation['h'] / 2],
                           axis='columns')
    annotation.columns = ['label', 'x1', 'y1', 'x2', 'y2']
    annotation[['x1', 'y1', 'x2', 'y2']] = annotation[[
        'x1', 'y1', 'x2', 'y2']].clip(0, 1)

    image = cv.imread(
        str(annotation_file.with_suffix('.jpg')), cv.IMREAD_COLOR)
    img_height, img_width = image.shape[:2]
    annotation_abs_coords = annotation.copy()
    annotation_abs_coords[['x1', 'x2']] *= img_width
    annotation_abs_coords[['y1', 'y2']] *= img_height

    left = annotation_abs_coords['x1'].min()
    left = round(clip((left - left * 0.1), 0, img_width))
    top = annotation_abs_coords['y1'].min()
    top = round(clip((top - top * 0.1), 0, img_height))
    right = annotation_abs_coords['x2'].max()
    right = round(clip((right + right * 0.1), 0, img_width))
    bottom = annotation_abs_coords['y2'].max()
    bottom = round(clip((bottom + bottom * 0.1), 0, img_height))

    image = image[top:bottom, left:right]
    img_height, img_width = image.shape[:2]
    cv.imwrite(str(annotation_file.with_suffix('.jpg')), image)

    annotation_abs_coords[['x1', 'x2']] -= left
    annotation_abs_coords[['y1', 'y2']] -= top

    annotation = pd.concat([annotation_abs_coords['label'],
                            (annotation_abs_coords['x1'] + (
                                annotation_abs_coords['x2'] - annotation_abs_coords['x1']) / 2) / img_width,
                            (annotation_abs_coords['y1'] + (
                                annotation_abs_coords['y2'] - annotation_abs_coords['y1']) / 2) / img_height,
                            (annotation_abs_coords['x2'] -
                             annotation_abs_coords['x1']) / img_width,
                            (annotation_abs_coords['y2'] - annotation_abs_coords['y1']) / img_height],
                           axis='columns')
    annotation.columns = ['label', 'cx', 'cy', 'w', 'h']

    annotation.to_csv(annotation_file, sep=' ',
                      line_terminator='\n', header=False, index=False)

image_names = obj_train_data_path.glob('*.jpg')
image_names = pd.DataFrame(image_names, columns=['filename'])
image_ids = image_names['filename'].map(
    lambda x: x.stem.split('_')[1]).astype(int)
image_ids.name = 'ids'
image_names = pd.concat([image_names, image_ids], axis='columns')
image_names['filename'] = image_names['filename'].map(
    lambda x: Path('data', x.relative_to(dataset_path)).as_posix())
image_names = image_names.set_index('ids')
image_names = image_names.sort_index()

train_set = image_names.sample(
    frac=trainset_percentage, random_state=args.random_state)
valid_set = image_names.loc[image_names.index.difference(train_set.index)]

train_set.to_csv(train_path, line_terminator='\n', header=False, index=False)
valid_set.to_csv(valid_path, line_terminator='\n', header=False, index=False)

pd.Series(classes_whitelist).to_csv(
    obj_names_path, line_terminator='\n', header=False, index=False)

obj_data = [
    f'classes = {len(classes_whitelist)}',
    'train = data/train.txt',
    'valid = data/valid.txt',
    'names = data/obj.names',
    'backup = backup/'
]

pd.Series(obj_data).to_csv(obj_data_path,
                           line_terminator='\n', header=False, index=False)
