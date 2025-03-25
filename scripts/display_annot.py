import argparse
import openslide
import cv2
import json
import numpy as np
from pathlib import Path
from constants import LYMPHOCYTE_SIZE_UM, MONOCYTE_SIZE_UM


def draw_patch(patch_path: Path) -> np.ndarray:
    """Function loads patch and draws white blood cell labels."""
    patch = cv2.imread(str(patch_path))
    with open(patch_path.parents[1] / 'labels' / patch_path.with_suffix('.txt').name) as label_file:
        lines = label_file.readlines()
        for line in lines:
            color = (245, 0, 203) if int(line[0]) == 0 else (0, 0, 229)
            line = np.array(line.strip().split()[1:]).astype(float)
            line = np.floor(line / [1, 1, 2, 2] * patch.shape[0]).astype(int)
            cv2.rectangle(patch, tuple(line[:2] - line[2:]), tuple(line[:2] + line[2:]), color=color, thickness=2)
    return patch


def draw_roi(image_path: Path, roi: dict, level: int) -> np.ndarray:
    """Function loads ROI and draws white blood cell annotations."""
    slide = openslide.OpenSlide(image_path)
    image_name = '_'.join(image_path.name.split('_')[:2])
    annot_path = image_path.parents[2] / 'annotations/json_pixel'

    left_top = np.floor(np.min(roi['polygon'], axis=0)).astype(int)
    polygon = np.array(roi['polygon']) / slide.level_downsamples[level]
    left = np.floor(np.min(polygon[:, 0])).astype(int)
    top = np.floor(np.min(polygon[:, 1])).astype(int)
    right = np.ceil(np.max(polygon[:, 0])).astype(int)
    bottom = np.ceil(np.max(polygon[:, 1])).astype(int)

    region = slide.read_region(left_top, level, (right - left + 1, bottom - top + 1))
    region = cv2.cvtColor(np.array(region), cv2.COLOR_RGBA2BGR)
    polygon = np.round(polygon - [left, top]).astype(int).reshape((-1, 1, 2))
    cv2.polylines(region, [polygon], isClosed=True, color=(223, 67, 3), thickness=2)

    lymph_size = LYMPHOCYTE_SIZE_UM / float(slide.properties['openslide.mpp-x']) / slide.level_downsamples[level]
    lymph_half_size = np.full(2, lymph_size / 2)
    with open(annot_path / (image_name + '_lymphocytes.json'), 'r') as lymph_file:
        annot = json.load(lymph_file)
        for cell in annot['points']:
            cell = np.array(cell['point']) / slide.level_downsamples[level]
            if left <= cell[0] < right + 1 and top <= cell[1] < bottom + 1:
                c1 = (cell - [left, top] - lymph_half_size).round().astype(int)
                c2 = (cell - [left, top] + lymph_half_size).round().astype(int)
                cv2.rectangle(region, np.maximum(c1, 0), np.minimum(c2, [right - left, bottom - top]),
                              color=(245, 0, 203), thickness=1)

    mono_size =  MONOCYTE_SIZE_UM / float(slide.properties['openslide.mpp-x']) / slide.level_downsamples[level]
    mono_half_size = np.full(2, mono_size / 2)
    with open(annot_path / (image_name + '_monocytes.json'), 'r') as mono_file:
        annot = json.load(mono_file)
        for cell in annot['points']:
            cell = np.array(cell['point']) / slide.level_downsamples[level]
            if left <= cell[0] < right + 1 and top <= cell[1] < bottom + 1:
                c1 = (cell - [left, top] - mono_half_size).round().astype(int)
                c2 = (cell - [left, top] + mono_half_size).round().astype(int)
                cv2.rectangle(region, np.maximum(c1, 0), np.minimum(c2, [right - left, bottom - top]),
                              color=(0, 0, 229), thickness=1)

    return region


def draw_wsi(image_path: Path, level: int) -> np.ndarray:
    """Functions loads WSI and draws all ROI borders."""
    if level < 2:
        raise RuntimeError('Level must be greater than 1 to fit image into memory')

    slide = openslide.OpenSlide(image_path)
    image_name = '_'.join(image_path.name.split('_')[:2])
    annot_path = image_path.parents[2] / 'annotations/json_pixel' / (image_name + '_inflammatory-cells.json')

    level = min(level, slide.level_count - 1)
    image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)
    with open(annot_path) as f:
        annot = json.load(f)
        for roi in annot['rois']:
            polygon = np.array(roi['polygon']) / slide.level_downsamples[level]
            polygon = np.round(polygon).astype(int).reshape((-1, 1, 2))
            cv2.polylines(image, [polygon], isClosed=True, color=(223, 67, 3), thickness=2)

    slide.close()
    return image


def display_wsi(image_path: Path, wsi_level: int, roi_level: int) -> int:
    """Function gradually displays WSI and its ROIs with annotations."""
    image_name = '_'.join(image_path.name.split('_')[:2])
    with open(image_path.parents[2] / 'annotations/json_pixel' / (image_name + '_inflammatory-cells.json')) as f:
        annot = json.load(f)

    cv2.imshow(image_path.name, draw_wsi(image_path, wsi_level))
    for roi in annot['rois']:
        image = draw_roi(image_path, roi, roi_level)
        key = cv2.waitKey()
        cv2.destroyAllWindows()
        if key == ord('q'):
            return key
        cv2.imshow(image_path.name + ' ' + roi['name'], image)

    key = cv2.waitKey()
    cv2.destroyAllWindows()
    return key


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script displays random or specified whole slide images or patches. '
                                                 'If a path to a directory is provided then a series of random images is displayed until terminated. '
                                                 'Press any key to show next image or press \'q\' to quit at any time. ')
    parser.add_argument('path', type=Path, help='Path to a directory, whole slide image or a patch')
    args = parser.parse_args()

    if args.path.is_dir():
        images = list(args.path.glob('*.tif')) + list(args.path.glob('*.png'))
        if len(images) == 0:
            raise RuntimeError(f'No supported images found in {args.path}')

        while True:
            path = np.random.choice(images)
            if path.suffix == '.tif':
                if display_wsi(path, wsi_level=7, roi_level=2) == ord('q'):
                    break
            else:
                cv2.imshow(path.name, draw_patch(path))
                k = cv2.waitKey()
                cv2.destroyAllWindows()
                if k == ord('q'):
                    break

    elif not args.path.exists():
        raise RuntimeError(f'File {args.path} does not exist')

    elif args.path.suffix == '.tif':
        display_wsi(args.path, wsi_level=7, roi_level=2)

    elif args.path.suffix == '.png':
        cv2.imshow(args.path.name, draw_patch(args.path))
        cv2.waitKey()
        cv2.destroyAllWindows()

    else:
        raise RuntimeError(f'Invalid file type: {args.path}')
