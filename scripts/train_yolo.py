import sys
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO, settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script trains a YOLO model')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--imgsz', type=int, required=True)
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--model', type=str, default='yolo11m.pt')
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()

    # Setting up project directory
    project_name = Path('yolo') / args.project
    run_name = f'img{args.imgsz}_ep{args.epochs}_{Path(args.model).with_suffix("").name}{args.name}'
    run_directory = project_name / run_name
    if run_directory.exists():
        print(f'YOLO run {run_directory} already exists')
        sys.exit()

    settings.update({'datasets_dir': 'data'})
    settings.update({'weights_dir': 'yolo/weights'})

    # Training YOLO
    iou = [0.7, 0.3, 0.1]
    train_cfg = {'data': args.data, 'epochs': args.epochs, 'imgsz': args.imgsz, 'project': project_name, 'name': run_name,
                 'deterministic': False, 'cos_lr': True, 'patience': args.epochs, 'workers': 0, 'cache': 'disk', 'resume': args.resume}
    aug_cfg = {'hsv_s': 0.4, 'degrees': 15, 'scale': 0, 'flipud': 0.5, 'mosaic': 0, 'erasing': 0, 'close_mosaic': 0}
    val_cfg = {'conf': 0.001, 'max_det': None, 'agnostic_nms': True, 'iou': iou[0]}
    model = YOLO(args.model)
    model.train(**train_cfg, **aug_cfg, **val_cfg)

    # Additional YOLO validation
    for x in iou[1:]:
        print(f'\nValidating yolo with iou={x}')
        val_cfg['iou'] = x
        model.val(**val_cfg, imgsz=args.imgsz, project=run_directory, name=f'iou{x}')

    # Directory files cleanup
    metric_files = list(run_directory.glob('*curve*')) + list(run_directory.glob('*matrix*'))
    (run_directory / f'iou{iou[0]}').mkdir(parents=False, exist_ok=False)
    for file in metric_files:
        shutil.move(file, run_directory / f'iou{iou[0]}')
    for file in run_directory.rglob('val_*jpg'):
        file.unlink()
