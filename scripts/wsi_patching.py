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


def write_yaml(output_dir):
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


def get_box(cell: tuple[float, float], cell_size: float, patch_size: int, class_label: int) -> str:
    """Function converts coordinates of a cell to basic box label."""
    x, y = cell
    box_width = (min(x + cell_size / 2, patch_size) - max(x - cell_size / 2, 0)) / patch_size
    box_height = (min(y + cell_size / 2, patch_size) - max(y - cell_size / 2, 0)) / patch_size
    x = (max(x - cell_size / 2, 0) / patch_size) + (box_width / 2)
    y = (max(y - cell_size / 2, 0) / patch_size) + (box_height / 2)
    return f'{class_label} {x} {y} {box_width} {box_height}\n'


def segment_labels(predictor: SAM2Predictor, cells: list[tuple[float, float]], cell_size: float, patch_size: int,
                   class_label: int) -> list[str]:
    """Function converts cell annotations to box labels created using Segment Anything Model 2."""
    global CELLS, NO_DETECTION, WRONG_DETECTION
    CELLS += len(cells)

    labels = []
    for x, y in cells:
        res = predictor(points=[x, y])[0].cpu().numpy()
        box = None
        for i, xywh in enumerate(res.boxes.xywh):
            if xywh[2] > SEG_MAX_CELL_SIZE_MULT * cell_size or xywh[3] > SEG_MAX_CELL_SIZE_MULT * cell_size:
                continue
            box = res.boxes.xywhn[i]

        if box is not None:
            box = ' '.join(box.astype(str))
            labels.append(f'{class_label} {box}\n')
        else:
            labels.append(get_box((x, y), cell_size, patch_size, class_label))
            if len(res) == 0:
                NO_DETECTION += 1
            else:
                WRONG_DETECTION += 1

    return labels


def patch_image(image_path: Path, patch_size: int, patch_dir: Path, labels_dir: Path, predictor: SAM2Predictor,
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
            patch.convert('RGB').save(patch_dir / patch_name)

            if predictor is None:
                labels = []
                for cell in lymph_split[i]:
                    labels.append(get_box(cell, lymph_size, patch_size, class_label=0))
                for cell in mono_split[i]:
                    labels.append(get_box(cell, mono_size, patch_size, class_label=1))
            else:
                predictor.set_image(str(patch_dir / patch_name))
                labels = segment_labels(predictor, lymph_split[i], lymph_size, patch_size, class_label=0)
                labels.extend(segment_labels(predictor, mono_split[i], mono_size, patch_size, class_label=1))
                predictor.reset_image()

            label_name = Path(patch_name).with_suffix('.txt')
            with open(labels_dir / label_name, 'x') as label_file:
                label_file.writelines(labels)

    slide.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script splits annotated regions of interest into patches of equal size')
    parser.add_argument('input_directory', type=Path, help='Directory containing whole slide images')
    parser.add_argument('output_directory', type=Path, help='Output directory where the dataset will be created')
    parser.add_argument('patch_size', type=int, help='Size of patch side in pixels')
    parser.add_argument('--segment', action='store_true', help='Enable cell segmentation to improve bounding boxes')
    args = parser.parse_args()

    input_dir = args.input_directory
    patch_directory = args.output_directory / (input_dir.name + str(args.patch_size)) / 'images'
    patch_directory.mkdir(parents=True, exist_ok=False)
    label_directory = args.output_directory / (input_dir.name + str(args.patch_size)) / 'labels'
    label_directory.mkdir(parents=True, exist_ok=False)

    model = None
    CELLS = NO_DETECTION = WRONG_DETECTION = 0
    if args.segment:
        SEG_MAX_CELL_SIZE_MULT = 1.5
        model = SAM2Predictor(overrides=dict(model='sam2.1_b.pt', save=False, verbose=False))
        model.get_model()

    image_cnt = len([x for x in input_dir.iterdir()])
    for wsi_path in tqdm(input_dir.iterdir(), total=image_cnt, desc=f'Patching images in {input_dir}'):
        patch_image(wsi_path, args.patch_size, patch_directory, label_directory, model, shift_num=4)

    if args.segment:
        print(f'Segmentation results:')
        print(f'Number of cells in dataset: {CELLS}')
        print(f'Cells not detected: {NO_DETECTION}')
        print(f'Wrong detections: {WRONG_DETECTION}')

    autosplit(patch_directory, (0.8, 0.2, 0))
    write_yaml(args.output_directory / (input_dir.name + str(args.patch_size)))
