import cv2
import re
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.engine.results import Results
from display_annot import draw_patch
from constants import LYMPHOCYTE_BGR, MONOCYTE_BGR


def draw_predictions_on(patch: np.ndarray, result: Results) -> np.ndarray:
    """Function draws input predictions on loaded patch."""
    boxes = result.cpu().numpy().boxes
    for i, box in enumerate(boxes.xyxy):
        color = LYMPHOCYTE_BGR if int(boxes.cls[i]) == 0 else MONOCYTE_BGR
        box = np.round(box).astype(int)
        cv2.rectangle(patch, tuple(box[:2]), tuple(box[2:]), color=color, thickness=2)
    return patch


def draw_predictions(patch_path: Path, result: Results) -> np.ndarray:
    """Function loads patch and draws input predictions."""
    patch = cv2.imread(str(patch_path))
    return draw_predictions_on(patch, result)


def compare_patches(patches: list[np.ndarray], title: str) -> int:
    """Function displays input patches next to each other."""
    rows = []
    img = None
    h_separator = np.full((patches[0].shape[0], 10, 3), 255, dtype=np.uint8)
    for i, patch in enumerate(patches):
        img = patch if img is None else cv2.hconcat([img, h_separator, patch])
        if (len(patches) != 4 and (i + 1) % 3 == 0) or (len(patches) == 4 and (i + 1) % 2 == 0):
            rows.append(img)
            img = None

    if img is not None:
        if len(rows) > 0:
            fill = np.full((img.shape[0], rows[0].shape[1] - img.shape[1], 3), 255, dtype=np.uint8)
            img = cv2.hconcat([img, fill])
        rows.append(img)

    img = rows[0]
    v_separator = np.full((10, rows[0].shape[1], 3), 255, dtype=np.uint8)
    for row in rows[1:]:
        img = cv2.vconcat([img, v_separator, row])

    cv2.imshow(title, img)
    key = cv2.waitKey()
    cv2.destroyAllWindows()
    return key


def get_patch_split(dataset: str) -> tuple[int, list[Path], list[Path]]:
    """Function determines patch size of input dataset and loads lists of train and validation patches separately."""
    imgsz = re.search(r'(?<!\d)\d{3}(?!\d)', dataset).group()
    with open(Path('data') / dataset / 'autosplit_train.txt', 'r') as train_file:
        train_images = []
        for line in train_file:
            train_images.append(Path('data') / dataset / line.strip())
    with open(Path('data') / dataset / 'autosplit_val.txt', 'r') as val_file:
        val_images = []
        for line in val_file:
            val_images.append(Path('data') / dataset / line.strip())
    return int(imgsz), train_images, val_images


def compare_boxes(datasets: list[str], draw_dots: bool = True) -> None:
    """Function compares different types of bounding box labels."""
    print('Comparing boxes:')
    for dataset in datasets:
        print(dataset)

    _, train_images, _ = get_patch_split(datasets[0])
    while True:
        patch_name = np.random.choice(train_images).name.replace('IHC', 'PAS')
        patches = []
        for dataset in datasets:
            patch_path = Path('data') / dataset / 'images' / patch_name
            if 'ihc' in dataset:
                patch_path = Path('data') / dataset / 'images' / patch_name.replace('PAS', 'IHC')
            patches.append(draw_patch(patch_path, draw_annot=draw_dots))
        key = compare_patches(patches, patch_name)
        if key == ord('q'):
            break


def compare_nms(dataset: str, model: str, nms: list[float], conf: float) -> None:
    """Function compares model predictions with different iou thresholds for non-maximum suppression."""
    print('Comparing non-maximum suppression thresholds:')
    for iou in nms:
        print(f'Predictions with iou={iou}')

    imgsz, train_images, val_images = get_patch_split(dataset)
    model = YOLO(model)
    while True:
        patch_path = np.random.choice(train_images + val_images)
        patches = []
        for iou in nms:
            results = model.predict(patch_path, imgsz=imgsz, conf=conf, max_det=None, agnostic_nms=True, iou=iou, verbose=False)
            pred_patch = draw_predictions_on(draw_patch(patch_path, draw_labels=False, draw_annot=True), results[0])
            patches.append(pred_patch)
        title = 'TRAIN ' + patch_path.name if patch_path in train_images else 'VAL ' + patch_path.name
        key = compare_patches(patches, title)
        if key == ord('q'):
            break


def compare_models(dataset: str, models: list[str], nms: float) -> None:
    """Function compares predictions of multiple models."""
    print('Comparing models:')
    for model_name in models:
        print(f'Model {model_name}')

    imgsz, train_images, val_images = get_patch_split(dataset)
    while True:
        patch_path = np.random.choice(train_images + val_images)
        patches = []
        for model_name in models:
            model = YOLO(model_name)
            results = model.predict(patch_path, imgsz=imgsz, conf=0.25, max_det=None, agnostic_nms=True, iou=nms, verbose=False)
            pred_patch = draw_predictions_on(draw_patch(patch_path, draw_labels=False, draw_annot=True), results[0])
            patches.append(pred_patch)
        title = 'TRAIN ' + patch_path.name if patch_path in train_images else 'VAL ' + patch_path.name
        key = compare_patches(patches, title)
        if key == ord('q'):
            break


if __name__ == '__main__':
    # Bounding boxe generation
    # compare_boxes(['basic_box/pas-cpg256', 'pure_seg_box/pas-cpg256'])
    # compare_boxes(['basic_box/pas-cpg512', 'pure_seg_box/pas-cpg512', 'pure_seg_box/ihc512'])
    # compare_boxes(['basic_box/pas-cpg256', 'pure_seg_box/pas-cpg256', 'pure_seg_box/pas-cpg256_pad5',
    #                'pure_seg_box/pas-cpg256_pad10'])

    # Experiment 1 - Labels
    compare_models('basic_box/pas-cpg256', ['yolo/basic_box/img256_ep100_yolo11m/weights/last.pt',
                                                    'yolo/pure_box/img256_ep100_yolo11m/weights/last.pt',
                                                    'yolo/seg_box/img256_ep100_yolo11m/weights/last.pt'], nms=0.3)

    # Experiment 2 - Padding
    # compare_models('basic_box/pas-cpg256', ['yolo/pure_box/img256_ep100_yolo11m/weights/last.pt',
    #                                                 'yolo/pure_box/padding5/img256_ep200_yolo11m/weights/last.pt',
    #                                                 'yolo/pure_box/padding10/img256_ep200_yolo11m/weights/last.pt',
    #                                                 'yolo/seg_box/img256_ep100_yolo11m/weights/last.pt',
    #                                                 'yolo/seg_box/padding5/img256_ep200_yolo11m/weights/last.pt',
    #                                                 'yolo/seg_box/padding10/img256_ep200_yolo11m/weights/last.pt'], nms=0.3)

    # Experiment 3 - NMS
    # compare_nms('basic_box/pas-cpg256', 'yolo/basic_box/img256_ep100_yolo11m/weights/last.pt', [0.7, 0.3, 0.1], conf=0.25)
    # compare_nms('basic_box/pas-cpg256', 'yolo/pure_box/img256_ep100_yolo11m/weights/last.pt', [0.7, 0.3, 0.1], conf=0.25)

    # Experiment 4 - IHC
    # compare_boxes(['basic_box/pas-cpg256', 'basic_box/ihc256'])
