import sys
import shutil
import argparse
import openslide
import json
import numpy as np
from pathlib import Path
from shapely import Polygon, Point
from tqdm import tqdm
from ultralytics.data.utils import autosplit
from ultralytics.models.sam import SAM2Predictor
from constants import LYMPHOCYTE_SIZE_UM, MONOCYTE_SIZE_UM


def write_yaml(output_dir: Path) -> None:
    """Function attempts to create dataset.yaml file."""
    if not output_dir.is_relative_to('data'):
        print('Output directory is not relative to dataset directory - no dataset.yaml will be created')
    else:
        with open(output_dir / 'dataset.yaml', 'x') as f:
            f.write(f"path: {output_dir.relative_to('data').as_posix()}\n")
            f.write('train: autosplit_train.txt\n')
            f.write('val: autosplit_val.txt\n')
            f.write('\nnames:\n')
            f.write('  0: lymphocyte\n')
            f.write('  1: monocyte\n')


def add_padding(input_dir: Path, output_dir: Path, padding: int, patch_size: int) -> None:
    """Function creates new dataset with added padding to bounding boxes."""
    print(f'Copying patches from {input_dir} to {output_dir}')
    shutil.copytree(input_dir / 'images', output_dir / 'images', copy_function=shutil.copy)
    shutil.copytree(input_dir / 'annotations', output_dir / 'annotations', copy_function=shutil.copy)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=False)
    norm_padding = padding / patch_size

    patch_cnt = len([x for x in (input_dir / 'labels').iterdir()])
    for label_path in tqdm((input_dir / 'labels').iterdir(), total=patch_cnt, desc=f'Padding boxes in {input_dir}'):
        with open(label_path, 'r') as label_file:
            labels = label_file.readlines()
        padded_labels = []
        for label in labels:
            label = np.array(label.strip().split(), dtype=float)
            cls, x_center, y_center, width, height = label
            new_width = (min(x_center + width / 2 + norm_padding, 1) - max(x_center - width / 2 - norm_padding, 0))
            new_height = (min(y_center + height / 2 + norm_padding, 1) - max(y_center - height / 2 - norm_padding, 0))
            new_x = max(x_center - width / 2 - norm_padding, 0) + (new_width / 2)
            new_y = max(y_center - height / 2 - norm_padding, 0) + (new_height / 2)
            padded_labels.append(f'{int(cls)} {new_x} {new_y} {new_width} {new_height}\n')

        new_label_path = output_dir / 'labels' / label_path.name
        with open(new_label_path, 'x') as new_label_file:
            new_label_file.writelines(padded_labels)


def filter_labels(autosplit_path: Path, pure: bool) -> tuple[int, int, int]:
    """Function filters segmented labels and can leave out basic boxes."""
    with open(autosplit_path, 'r') as autosplit_file:
        images = autosplit_file.readlines()

    cell_num = no_detection = big_detection = 0
    for image in images:
        label_path = autosplit_path.parent / 'labels' / Path(image).with_suffix('.txt').name
        with open(label_path, 'r') as label_file:
            labels = label_file.readlines()

        new_labels = []
        for label in labels:
            arr = label.strip().split()
            if len(arr) == 6:
                if not pure:
                    new_labels.append(' '.join(arr[1:]) + '\n')
                if arr[0] == 'NONE':
                    no_detection += 1
                elif arr[0] == 'BIG':
                    big_detection += 1
            else:
                new_labels.append(label)
            cell_num += 1

        with open(label_path, 'w') as label_file:
            label_file.writelines(new_labels)

    return cell_num, no_detection, big_detection


def get_coverage(roi: dict, patch_size: int, shift_num: int) -> list[tuple[int, int]]:
    """Function finds coverage of ROI by patches of same size that are entirely contained within the region."""
    polygon = Polygon(roi['polygon'])
    left, top, right, bottom = polygon.bounds

    left = np.floor(left).astype(int)
    top = np.floor(top).astype(int)
    right = np.ceil(right).astype(int)
    bottom = np.ceil(bottom).astype(int)

    best_coverage = []
    for x_shift in range(0, patch_size, patch_size//shift_num):
        for y_shift in range(0, patch_size, patch_size//shift_num):

            coverage = []
            for x in range(left + x_shift, right - patch_size + 2, patch_size):
                for y in range(top + y_shift, bottom - patch_size + 2, patch_size):
                    covered = True
                    for (corner_x, corner_y) in [(x, y), (x + patch_size - 1, y), (x, y + patch_size - 1),
                                                 (x + patch_size - 1, y + patch_size - 1)]:
                        if not polygon.contains(Point(corner_x, corner_y)):
                            covered = False
                            break
                    if covered:
                        coverage.append((x, y))
            if len(coverage) > len(best_coverage):
                best_coverage = coverage

    return best_coverage


def split_cells(annot: dict, coverage: list[tuple[int, int]], patch_size: int) -> list[list[tuple[float, float]]]:
    """Function divides cell annotations among created patches."""
    cells = [[] for _ in range(len(coverage))]
    for cell in annot['points']:
        cell = np.array(cell['point'])
        for i, (x, y) in enumerate(coverage):
            if x <= cell[0] < x + patch_size and y <= cell[1] < y + patch_size:
                cells[i].append((cell[0] - x, cell[1] - y))
                break
    return cells


def get_basic_box(cell: tuple[float, float], cell_size: float, patch_size: int, class_label: int) -> str:
    """Function converts coordinates of a cell to basic box label."""
    x, y = cell
    box_width = (min(x + cell_size / 2, patch_size) - max(x - cell_size / 2, 0)) / patch_size
    box_height = (min(y + cell_size / 2, patch_size) - max(y - cell_size / 2, 0)) / patch_size
    x = (max(x - cell_size / 2, 0) / patch_size) + (box_width / 2)
    y = (max(y - cell_size / 2, 0) / patch_size) + (box_height / 2)
    return f'{class_label} {x} {y} {box_width} {box_height}\n'


def segment_labels(predictor: SAM2Predictor, cells: list[tuple[float, float]], cell_size: float, patch_size: int,
                   seg_size_mult: float, class_label: int) -> list[str]:
    """Function converts cell annotations to box labels created using SAM 2."""
    labels = []
    for x, y in cells:
        res = predictor(points=[x, y])[0].cpu().numpy()
        label = None
        for i, xywh in enumerate(res.boxes.xywh):
            if xywh[2] > seg_size_mult * cell_size or xywh[3] > seg_size_mult * cell_size:
                continue
            box = ' '.join(res.boxes.xywhn[i].astype(str))
            label = f'{class_label} {box}\n'
            break

        if label is None:
            label = get_basic_box((x, y), cell_size, patch_size, class_label)
            if len(res) == 0:
                label = 'NONE ' + label
            else:
                label = 'BIG ' + label
        labels.append(label)
    return labels


def patch_image(image_path: Path, patch_size: int, output_dir: Path, predictor: SAM2Predictor, seg_size_mult: float,
                shift_num: int) -> None:
    """Function splits all ROIs of a WSI into patches and saves them and their annotations."""
    slide = openslide.OpenSlide(image_path)
    image_name = '_'.join(image_path.name.split('_')[:2])
    annot_path = image_path.parents[2] / 'annotations/json_pixel'

    lymph_size = LYMPHOCYTE_SIZE_UM / float(slide.properties['openslide.mpp-x'])
    mono_size = MONOCYTE_SIZE_UM / float(slide.properties['openslide.mpp-x'])

    with open(annot_path / (image_name + '_lymphocytes.json'), 'r') as lymph_file:
        lymph_annot = json.load(lymph_file)
    with open(annot_path / (image_name + '_monocytes.json'), 'r') as mono_file:
        mono_annot = json.load(mono_file)

    for roi in lymph_annot['rois']:
        coverage = get_coverage(roi, patch_size, shift_num)
        lymph_split = split_cells(lymph_annot, coverage, patch_size)
        mono_split = split_cells(mono_annot, coverage, patch_size)
        for i, (x, y) in enumerate(coverage):
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch_name = image_path.with_suffix('').name + f"_{roi['name'].replace(' ', '')}_{i}.png"
            patch.convert('RGB').save(output_dir / 'images' / patch_name)

            if predictor is None:
                labels = []
                for cell in lymph_split[i]:
                    labels.append(get_basic_box(cell, lymph_size, patch_size, class_label=0))
                for cell in mono_split[i]:
                    labels.append(get_basic_box(cell, mono_size, patch_size, class_label=1))
            else:
                predictor.set_image(str(output_dir / 'images' / patch_name))
                labels = segment_labels(predictor, lymph_split[i], lymph_size, patch_size, seg_size_mult, class_label=0)
                labels.extend(segment_labels(predictor, mono_split[i], mono_size, patch_size, seg_size_mult, class_label=1))
                predictor.reset_image()

            label_name = Path(patch_name).with_suffix('.txt')
            with open(output_dir / 'labels' / label_name, 'x') as label_file:
                label_file.writelines(labels)

            with open(output_dir / 'annotations' / label_name, 'x') as annot_file:
                for cell_x, cell_y in lymph_split[i]:
                    annot_file.write(f'0 {cell_x} {cell_y}\n')
                for cell_x, cell_y in mono_split[i]:
                    annot_file.write(f'1 {cell_x} {cell_y}\n')

    slide.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script splits annotated regions of interest into patches of equal size')
    parser.add_argument('input_directory', type=Path, help='Directory containing whole slide images')
    parser.add_argument('output_directory', type=Path, help='Output directory where the dataset will be created')
    parser.add_argument('patch_size', type=int, help='Size of patch side in pixels')
    parser.add_argument('--segment', action='store_true', help='Enable cell segmentation to improve bounding boxes')
    parser.add_argument('--pure', action='store_true', help='Use only segmented boxes for training data')
    parser.add_argument('--padding', type=int, help='Number of pixels used to pad bounding boxes. With this option '
                        'enabled, the input_directory should already contain patched images and this script will only adjust the labels')
    args = parser.parse_args()

    if args.padding is not None:
        output_directory = args.output_directory / (args.input_directory.name + '_pad' + str(args.padding))
        add_padding(args.input_directory, output_directory, args.padding, args.patch_size)
        autosplit(output_directory / 'images', (0.8, 0.2, 0))
        write_yaml(output_directory)
        sys.exit()

    output_directory = args.output_directory / (args.input_directory.name + str(args.patch_size))
    (output_directory / 'images').mkdir(parents=True, exist_ok=False)
    (output_directory / 'labels').mkdir(parents=True, exist_ok=False)
    (output_directory / 'annotations').mkdir(parents=True, exist_ok=False)

    model = None
    segmentation_mult = 1.5
    if args.segment:
        model = SAM2Predictor(overrides=dict(model='sam2.1_b.pt', save=False, verbose=False))
        model.get_model()

    image_cnt = len([x for x in args.input_directory.iterdir()])
    for wsi_path in tqdm(args.input_directory.iterdir(), total=image_cnt, desc=f'Patching images in {args.input_directory}'):
        patch_image(wsi_path, args.patch_size, output_directory, model, segmentation_mult, shift_num=4)

    autosplit(output_directory / 'images', (0.8, 0.2, 0))
    write_yaml(output_directory)

    if args.segment:
        train_res = filter_labels(output_directory / 'autosplit_train.txt', args.pure)
        val_res = filter_labels(output_directory / 'autosplit_val.txt', False)
        result = tuple(x + y for x, y in zip(train_res, val_res))
        print(f'Segmentation results:')
        print(f'Number of cells in dataset: {result[0]}')
        print(f'Cells not detected: {result[1]}')
        print(f'Wrong detections: {result[2]}')
