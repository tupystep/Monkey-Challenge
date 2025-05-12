import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
from scipy.spatial.distance import cdist
from sklearn.metrics import auc
from ultralytics import YOLO
from constants import IMAGE_MPP


def average_curves(xs: list[np.ndarray], ys: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate and average input curves with different numbers of points."""
    if xs[0][0] > xs[0][-1]:
        xs = [x[::-1] for x in xs]
        ys = [y[::-1] for y in ys]

    x_min = min(x.min() for x in xs)
    x_max = max(x.max() for x in xs)
    x_common = np.linspace(x_min, x_max, 1000)

    y_interp = np.array([np.interp(x_common, x, y) for x, y in zip(xs, ys)])
    y_avg = np.mean(y_interp, axis=0)
    return x_common, y_avg


def plot_p_curves(metrics: dict, save_dir: Union[Path, None]) -> None:
    """Function plots precision-confidence curves."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for precision, confidence, name in zip(metrics['precisions'], metrics['confidences'], metrics['names']):
        ax.plot(confidence, precision, linewidth=1, label=f'{name}')

    avg_confidence, avg_precision = average_curves(metrics['confidences'][:-1], metrics['precisions'][:-1])
    ax.plot(avg_confidence, avg_precision, linewidth=3, color='blue', label='class average')

    ax.set(title='Precision-Confidence Curve', xlabel='Confidence', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    if save_dir is not None:
        fig.savefig(save_dir / 'P_curve.png', dpi=250)
        plt.close(fig)


def plot_r_curves(metrics: dict, save_dir: Union[Path, None]) -> None:
    """Function plots recall-confidence curves."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    for recall, confidence, name in zip(metrics['recalls'], metrics['confidences'], metrics['names']):
        ax.plot(confidence, recall, linewidth=1, label=f'{name}')

    avg_confidence, avg_recall = average_curves(metrics['confidences'][:-1], metrics['recalls'][:-1])
    ax.plot(avg_confidence, avg_recall, linewidth=3, color='blue', label='class average')

    ax.set(title='Recall-Confidence Curve', xlabel='Confidence', ylabel='Recall', xlim=(0, 1), ylim=(0, 1))
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    if save_dir is not None:
        fig.savefig(save_dir / 'R_curve.png', dpi=250)
        plt.close(fig)


def plot_f1_curves(metrics: dict, save_dir: Union[Path, None]) -> None:
    """Function plots f1-confidence curves."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    f1_curves = []
    for recall, precision, confidence, name in zip(metrics['recalls'], metrics['precisions'], metrics['confidences'],
                                                   metrics['names']):
        f1 = 2 * (precision * recall) / (precision + recall + 0.000001)
        ax.plot(confidence, f1, linewidth=1, label=f'{name} {np.max(f1):.3f} at {confidence[np.argmax(f1)]:.3f}')
        f1_curves.append(f1)

    avg_confidence, avg_f1 = average_curves(metrics['confidences'][:-1], f1_curves[:-1])
    ax.plot(avg_confidence, avg_f1, linewidth=3, color='blue',
            label=f'class average {np.max(avg_f1):.3f} at {avg_confidence[np.argmax(avg_f1)]:.3f}')

    ax.set(title='F1-Confidence Curve', xlabel='Confidence', ylabel='F1', xlim=(0, 1), ylim=(0, 1))
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    if save_dir is not None:
        fig.savefig(save_dir / 'F1_curve.png', dpi=250)
        plt.close(fig)


def plot_pr_curves(metrics: dict, save_dir: Union[Path, None]) -> None:
    """Function plots precision-recall curves."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    aps = []
    for precision, recall, name in zip(metrics['precisions'], metrics['recalls'], metrics['names']):
        precision = np.append(precision, 0)
        recall = np.append(recall, 1)
        ap = auc(recall, precision)
        ax.plot(recall, precision, linewidth=1, label=f'{name} {ap:.3f}')
        aps.append(ap)

    avg_recall, avg_precision = average_curves(metrics['recalls'][:-1], metrics['precisions'][:-1])
    ax.plot(np.append(avg_recall, 1), np.append(avg_precision, 0), linewidth=3, color='blue',
            label=f'class average {np.mean(aps[:-1]):.3f} mAP')

    ax.set(title='Precision-Recall Curve', xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1))
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    if save_dir is not None:
        fig.savefig(save_dir / 'PR_curve.png', dpi=250)
        plt.close(fig)


def plot_froc_curves(metrics: dict, save_dir: Union[Path, None]) -> None:
    """Function plots free response operating characteristic curves."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    eval_thresholds = [10, 20, 50, 100, 200, 300]

    froc_scores = []
    for recall, fp_mm2, name in zip(metrics['recalls'], metrics['fp_mm2'], metrics['names']):
        froc = np.mean(np.interp(eval_thresholds, fp_mm2, recall))
        ax.plot(fp_mm2, recall, linewidth=1, label=f'{name} {froc:.3f}')
        froc_scores.append(froc)

    avg_fp_mm2, avg_recall = average_curves(metrics['fp_mm2'][:-1], metrics['recalls'][:-1])
    ax.plot(avg_fp_mm2, avg_recall, linewidth=3, color='blue',
            label=f'class average {np.mean(froc_scores[:-1]):.3f} FROC Score')

    ax.set(title='FROC Curve', xlabel='False Positives per mm2', ylabel='Recall', xlim=(0, 500), ylim=(0, 1))
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    if save_dir is not None:
        fig.savefig(save_dir / 'FROC_curve.png', dpi=250)
        plt.close(fig)


def plot_confusion_matrix(ground_truth: list[np.ndarray], gt_class: list[np.ndarray], preds: list[np.ndarray],
                          pred_class: list[np.ndarray], save_dir: Union[Path, None]) -> None:
    """Function creates and plots confusion matrix."""
    matrix = np.zeros((3, 3))
    for patch_gt, patch_gt_class, patch_preds, patch_pred_class in zip(ground_truth, gt_class, preds, pred_class):
        patch_gt_class = patch_gt_class.astype(int)
        patch_pred_class = patch_pred_class.astype(int)

        if len(patch_gt) == 0 or len(patch_preds) == 0:
            np.add.at(matrix, (-1, patch_gt_class), 1)
            np.add.at(matrix, (patch_pred_class, -1), 1)
            continue

        dist_matrix = cdist(patch_gt, patch_preds, 'euclidean')
        matched_gt = np.zeros(len(patch_gt), dtype=bool)
        matched_preds = np.zeros(len(patch_preds), dtype=bool)

        while np.min(dist_matrix) <= 5 / IMAGE_MPP:
            gt_idx, pred_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            if patch_gt_class[gt_idx] == 0 and dist_matrix[gt_idx, pred_idx] > 4 / IMAGE_MPP:
                dist_matrix[gt_idx, :] = np.inf
                continue

            matched_gt[gt_idx] = True
            matched_preds[pred_idx] = True
            matrix[patch_pred_class[pred_idx], patch_gt_class[gt_idx]] += 1

            dist_matrix[gt_idx, :] = np.inf
            dist_matrix[:, pred_idx] = np.inf

        np.add.at(matrix, (-1, patch_gt_class[~matched_gt]), 1)
        np.add.at(matrix, (patch_pred_class[~matched_preds], -1), 1)
    matrix[matrix < 0.5] = np.nan

    tick_labels = ['lymphocytes', 'monocytes', 'background']
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True, facecolor=(1, 1, 1))
    sns.heatmap(matrix, ax=ax, annot=True, cmap='Blues', fmt='.0f', annot_kws={"size": 8},
                xticklabels=tick_labels, yticklabels=tick_labels, square=True, vmin=0)
    ax.set(title='Confusion Matrix', xlabel='True', ylabel='Predicted')
    if save_dir is not None:
        fig.savefig(save_dir / 'confusion_matrix.png', dpi=250)
        plt.close(fig)


def match_predictions(ground_truth: np.ndarray, preds: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Function matches predictions to ground truth and returns TP and FP indices."""
    if len(ground_truth) == 0 or len(preds) == 0:
        return np.array([]), np.arange(len(preds))

    dist_matrix = cdist(ground_truth, preds, 'euclidean')
    matched_preds = np.zeros(len(preds), dtype=bool)
    while np.min(dist_matrix) <= margin:
        gt_idx, pred_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        matched_preds[pred_idx] = True

        dist_matrix[gt_idx, :] = np.inf
        dist_matrix[:, pred_idx] = np.inf

    tp_idx = np.nonzero(matched_preds)[0]
    fp_idx = np.nonzero(~matched_preds)[0]
    return tp_idx, fp_idx


def calculate_metrics(ground_truth: list[np.ndarray], preds: list[np.ndarray], conf: list[np.ndarray], patch_size: int,
                      margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Function calculates precision, recall and number of false positives per mm2 for each confidence threshold."""
    tp_conf = fp_conf = np.array([])
    for patch_gt, patch_preds, patch_conf in zip(ground_truth, preds, conf):
        tp_idx, fp_idx = match_predictions(patch_gt, patch_preds, margin)
        if len(tp_idx) > 0:
            tp_conf = np.concatenate([tp_conf, patch_conf[tp_idx]])
        if len(fp_idx) > 0:
            fp_conf = np.concatenate([fp_conf, patch_conf[fp_idx]])

    total_gt = np.sum([len(gt) for gt in ground_truth])
    area_mm2 = (patch_size*IMAGE_MPP) * (patch_size*IMAGE_MPP) * len(ground_truth) / 1000000
    thresholds = np.unique(np.concatenate([tp_conf, fp_conf]))
    thresholds = np.append(thresholds, [thresholds[-1] + 0.001, 1])[::-1]

    precision = []
    recall = []
    fp_mm2 = []
    for threshold in thresholds:
        tp = np.sum(tp_conf >= threshold)
        fp = np.sum(fp_conf >= threshold)

        precision.append(1.0 if (tp + fp) == 0 else tp / (tp + fp))
        recall.append(tp / total_gt)
        fp_mm2.append(fp / area_mm2)

    return np.array(precision), np.array(recall), np.array(fp_mm2), thresholds


def eval_model(model: YOLO, dataset: Path, pred_arguments: dict, save_dir: Union[Path, None]) -> None:
    """Function calculates and plots all point-wise evaluation metrics for input model on input dataset."""
    save_dir.mkdir(exist_ok=False)

    # Loading annotations
    ground_truth = []
    gt_class = []
    with open(dataset / 'autosplit_val.txt', 'r') as val_file:
        for line in val_file:
            annot_path = dataset / 'annotations' / Path(line.strip()).with_suffix('.txt').name
            if annot_path.stat().st_size == 0:
                ground_truth.append(np.empty((0, 2), dtype=np.float32))
                gt_class.append(np.empty(0, dtype=np.float32))
            else:
                annot = np.loadtxt(annot_path, dtype=np.float32, ndmin=2)
                ground_truth.append(annot[:, 1:])
                gt_class.append(annot[:, 0])

    # Getting predictions
    results = model.predict(dataset / 'autosplit_val.txt', **pred_arguments, verbose=False)
    preds = [res.cpu().numpy().boxes.xywh[:, :2] for res in results]
    conf = [res.cpu().numpy().boxes.conf for res in results]
    pred_class = [res.cpu().numpy().boxes.cls for res in results]

    # Calculating metrics for different classes
    metrics = {'precisions': [], 'recalls': [], 'fp_mm2': [], 'confidences': [], 'names': []}
    eval_configs = [
        {'name': 'lymphocytes', 'class_id': 0, 'margin': 4 / IMAGE_MPP},
        {'name': 'monocytes', 'class_id': 1, 'margin': 5 / IMAGE_MPP},
        {'name': 'combined cells', 'class_id': None, 'margin': 5 / IMAGE_MPP}
    ]
    for cfg in eval_configs:
        if cfg['class_id'] is not None:
            class_gt = [gt[gt_cls == cfg['class_id']] for gt, gt_cls in zip(ground_truth, gt_class)]
            class_preds = [pred[pred_cls == cfg['class_id']] for pred, pred_cls in zip(preds, pred_class)]
            class_conf = [cf[pred_cls == cfg['class_id']] for cf, pred_cls in zip(conf, pred_class)]
            precision, recall, fp_mm2, confidence = calculate_metrics(
                class_gt, class_preds, class_conf, patch_size=pred_arguments['imgsz'], margin=cfg['margin']
            )
        else:
            precision, recall, fp_mm2, confidence = calculate_metrics(
                ground_truth, preds, conf, patch_size=pred_arguments['imgsz'], margin=cfg['margin']
            )

        metrics['precisions'].append(precision)
        metrics['recalls'].append(recall)
        metrics['fp_mm2'].append(fp_mm2)
        metrics['confidences'].append(confidence)
        metrics['names'].append(cfg['name'])

    # Plotting metric curves
    plot_froc_curves(metrics, save_dir)
    plot_pr_curves(metrics, save_dir)
    plot_f1_curves(metrics, save_dir)
    plot_p_curves(metrics, save_dir)
    plot_r_curves(metrics, save_dir)

    # Plotting confusion matrix
    matrix_preds = [pred[cf >= 0.25] for pred, cf in zip(preds, conf)]
    matrix_pred_class = [pred_cls[cf >= 0.25] for pred_cls, cf in zip(pred_class, conf)]
    plot_confusion_matrix(ground_truth, gt_class, matrix_preds, matrix_pred_class, save_dir)


if __name__ == '__main__':
    import re
    from tqdm import tqdm

    models = Path('yolo').rglob('img*ep*yolo*')
    for model_name in tqdm(list(models)):
        imgsz = int(re.search(r'img(\d{3})_', str(model_name)).group(1))
        staining = 'ihc' if 'ihc' in str(model_name) else 'pas-cpg'
        if 'basic_box' in str(model_name):
            box = 'basic_box'
        elif 'seg_box' in str(model_name):
            box = 'seg_box'
        elif 'pure_box' in str(model_name):
            box = 'pure_seg_box'
        else:
            raise ValueError('Invalid model name')

        dataset_path = Path(f'data/{box}/{staining}{imgsz}')
        if imgsz == 640:
            p_size = int(re.search(r'yolo11m_p(\d{3})', str(model_name)).group(1))
            dataset_path = Path(f'data/{box}/{staining}{p_size}')

        for iou in [0.7, 0.3, 0.1]:
            yolo_model = YOLO(model_name / 'weights' / 'last.pt')
            val_cfg = {'imgsz': imgsz, 'batch': 16, 'conf': 0.001, 'max_det': None, 'agnostic_nms': True, 'iou': iou}
            eval_model(model=yolo_model, dataset=dataset_path, pred_arguments=val_cfg, save_dir=model_name / f'point_nms{iou}')
