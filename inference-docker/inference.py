import json
import openslide
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

IMAGE_MPP = 0.24199951445730394


def write_preds(output_path: Path, preds: list[list[float]], confidences: list[float], classes: list[int], patch_num: int):
    lymphocytes = {'name': 'lymphocytes', 'type': 'Multiple points', 'version': {'major': 1, 'minor': 0}, 'points': []}
    monocytes = {'name': 'monocytes', 'type': 'Multiple points', 'version': {'major': 1, 'minor': 0}, 'points': []}
    cells = {'name': 'inflammatory-cells', 'type': 'Multiple points', 'version': {'major': 1, 'minor': 0}, 'points': []}

    l = m = 1
    for (x, y), conf, cls in zip(preds, confidences, classes):
        cells['points'].append({'name': f'Point {l + m - 1}', 'point': [x, y, IMAGE_MPP], 'probability': conf})
        if cls == 0:
            lymphocytes['points'].append({'name': f'Point {l}', 'point': [x, y, IMAGE_MPP], 'probability': conf})
            l += 1
        if cls == 1:
            monocytes['points'].append({'name': f'Point {m}', 'point': [x, y, IMAGE_MPP], 'probability': conf})
            m += 1

    print(f'Detected {len(lymphocytes["points"])} lymphocytes and {len(monocytes["points"])} monocytes in {patch_num} patches')

    with open(output_path / 'detected-lymphocytes.json', 'x') as f:
        json.dump(lymphocytes, f, indent=4)

    with open(output_path / 'detected-monocytes.json', 'x') as f:
        json.dump(monocytes, f, indent=4)

    with open(output_path / 'detected-inflammatory-cells.json', 'x') as f:
        json.dump(cells, f, indent=4)


def predict_wsi(model: YOLO, pred_arguments: dict, image_path: Path, mask_path: Path, patch_size: int):
    image_slide = openslide.OpenSlide(image_path)
    mask_slide = openslide.OpenSlide(mask_path)

    preds = []
    confidences = []
    classes = []

    patch_num = 0
    patches = [(x, y) for x in range(0, mask_slide.dimensions[0] - patch_size + 1, patch_size) for y in
               range(0, mask_slide.dimensions[1], patch_size)]

    for x, y in tqdm(patches, desc=f'Detecting in image {image_path.name}'):
        tissue_mask = mask_slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
        if np.array(tissue_mask).any():
            patch_image = image_slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
            results = model.predict(patch_image, **pred_arguments, verbose=True)
            boxes = results[0].cpu().numpy().boxes

            preds.extend(((boxes.xywh[:, :2] + [x, y]) * IMAGE_MPP / 1000).tolist())
            confidences.extend(boxes.conf.tolist())
            classes.extend(boxes.cls.tolist())
            patch_num += 1

    image_slide.close()
    mask_slide.close()
    return preds, confidences, classes, patch_num


if __name__ == '__main__':
    input_image = Path('/input/images/kidney-transplant-biopsy-wsi-pas').glob('*.tif')
    input_image = list(input_image)[0]
    input_mask = Path('/input/images/tissue-mask').glob('*.tif')
    input_mask = list(input_mask)[0]

    output_dir = Path('/output')
    yolo_model = YOLO('/opt/ml/model/yolo11m_basicBox_img512.pt')

    patch = 512
    nms = 0.3
    pred_cfg = {'imgsz': patch, 'conf': 0.001, 'max_det': None, 'agnostic_nms': True, 'iou': nms}

    output = predict_wsi(yolo_model, pred_cfg, input_image, input_mask, patch)
    write_preds(output_dir, *output)
