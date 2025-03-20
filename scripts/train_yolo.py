import argparse
from pathlib import Path
from ultralytics import YOLO, settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script trains a YOLO model')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--imgsz', type=int, required=True)
    parser.add_argument('--project', type=str, default='')
    parser.add_argument('--model', type=str, default='yolo11m.pt')
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()

    project_name = Path('yolo') / args.project / Path(args.model).with_suffix('').name
    run_name = f'img{args.imgsz}_ep{args.epochs}'

    settings.update({'datasets_dir': 'data'})
    settings.update({'weights_dir': 'yolo/weights'})

    model = YOLO(args.model)
    results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, project=project_name, name=run_name,
                          resume=args.resume, workers=0, cos_lr=True)
