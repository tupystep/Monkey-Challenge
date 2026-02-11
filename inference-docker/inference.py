import json
import openslide
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.utils.nms import TorchNMS
from torch import from_numpy

IMAGE_MPP = 0.24199951445730394


def write_preds(output_path: Path, preds: Boxes):
    """Function writes input predictions as a JSON."""
    lymphocytes = {'name': 'lymphocytes', 'type': 'Multiple points', 'version': {'major': 1, 'minor': 0}, 'points': []}
    monocytes = {'name': 'monocytes', 'type': 'Multiple points', 'version': {'major': 1, 'minor': 0}, 'points': []}
    cells = {'name': 'inflammatory-cells', 'type': 'Multiple points', 'version': {'major': 1, 'minor': 0}, 'points': []}

    l = m = 1
    for box in preds:
        x, y = (box.xywh[0, :2] * IMAGE_MPP / 1000).tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        cells['points'].append({'name': f'Point {l + m - 1}', 'point': [x, y, IMAGE_MPP], 'probability': conf})
        if cls == 0:
            lymphocytes['points'].append({'name': f'Point {l}', 'point': [x, y, IMAGE_MPP], 'probability': conf})
            l += 1
        if cls == 1:
            monocytes['points'].append({'name': f'Point {m}', 'point': [x, y, IMAGE_MPP], 'probability': conf})
            m += 1

    with open(output_path / 'detected-lymphocytes.json', 'x') as f:
        json.dump(lymphocytes, f, indent=4)

    with open(output_path / 'detected-monocytes.json', 'x') as f:
        json.dump(monocytes, f, indent=4)

    with open(output_path / 'detected-inflammatory-cells.json', 'x') as f:
        json.dump(cells, f, indent=4)


def predict_wsi(model: YOLO, pred_arguments: dict, image_path: Path, mask_path: Path, patch_size: int):
    """Function uses input YOLO model to detect cells in the WSI ROI defined by the input tissue mask."""
    image_slide = openslide.OpenSlide(image_path)
    mask_slide = openslide.OpenSlide(mask_path)

    patch_num = 0
    patches = [(x, y) for x in range(0, mask_slide.dimensions[0] - patch_size + 1, patch_size // 2) for y in
               range(0, mask_slide.dimensions[1] - patch_size + 1, patch_size // 2)]

    preds = []
    for x, y in tqdm(patches, desc=f'Detecting in image {image_path.name}'):
        tissue_mask = mask_slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
        tissue_mask = np.array(tissue_mask)
        if tissue_mask.any():
            patch_image = image_slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
            results = model.predict(patch_image, **pred_arguments, verbose=True)
            boxes = results[0].cpu().numpy().boxes

            # Filter out predictions clipped to borders
            boxes = boxes[((boxes.xyxy[:, :2] > 3) & (boxes.xyxy[:, 2:] < (patch_size - 3))).all(axis=1)]
            # Filter out predictions outside of tissue mask
            centers = np.round(boxes.xywh[:, :2]).astype(int)
            boxes = boxes[tissue_mask[centers[:, 1], centers[:, 0]].any(axis=1)]
            # Change coordinate system to WSI
            boxes = Boxes(boxes.data + [x, y, x, y, 0, 0], orig_shape=boxes.orig_shape)

            preds.append(boxes)
            patch_num += 1
    preds = np.concatenate([pred_boxes.data for pred_boxes in preds], axis=0)
    preds = Boxes(preds, orig_shape=(patch_size, patch_size))

    image_slide.close()
    mask_slide.close()

    # Additional NMS
    nms_idx = TorchNMS.nms(from_numpy(preds.xyxy), from_numpy(preds.conf), pred_arguments['iou'])
    preds = preds[nms_idx]
    print(f'Detected {sum(preds.cls == 0)} lymphocytes and {sum(preds.cls == 1)} monocytes in {patch_num} patches')
    return preds


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

    predictions = predict_wsi(yolo_model, pred_cfg, input_image, input_mask, patch)
    write_preds(output_dir, predictions)
