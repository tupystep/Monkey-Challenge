# Detection of Inflammatory Cells in Histological Images

This repository contains a deep learning approach for the detection of
lymphocytes and monocytes in kidney transplant biopsies. It utilizes the YOLO
detection network implemented in the Ultralytics package and an annotated
histological dataset provided by the MONKEY challenge. The approach consists of
preprocessing Whole-Slide Images (WSIs) from the provided  dataset into patches,
converting dot annotations into bounding box labels, and training YOLO11.

## Scripts
The folder `scripts` contains the implementation of this pipeline. Most scripts
are modifiable using CLI arguments, with `-h` printing their description and
available arguments and their usage.

#### Training script `train_yolo.py`
`python train_yolo.py --data=data/patched_dataset/dataset.yaml --epochs=100
--imgsz=512`

#### Preprocessing script `wsi_patching.py`
`python wsi_patching.py data/raw/images/pas-cpg data/basic_box512 512`

#### Data vsualization script `display_annot.py`
`python display_annot.py data/raw/images/pas-cpg`

Evaluation script `evaluate.py` evaluates all trained models with their runs and
weights in the `yolo` directory. Script `comparison_util.py` displays various
comparisons between label types or model predictions - must be modified in code.

### Training runs
Directory `yolo` contains all training runs of created models. Model weights are
not available in the repository and only the best performing model is stored in 
`model.pt`. Folder `basic_box` contains BasiBox runs, folder `pure_box` contains
SegBox runs, and folder `seg_box` contains MixedBox runs as they are defined in
the thesis text. The runs contain loss functions and many visualized metrics
evaluated box-wise or point-wise at different NMS thresholds (0.7, 0.3, 0.1)

### Inference Docker
Code for the creation and saving of docker performing inference on WSIs. The
Dockerfile and scripts were provided by the MONKEY challenge and subsequently
modified. To run the docker locally the folder `inference-docker` must contain
the folders `test/input/images/kideny-transplant-biopsy-wsi-pas` with one WSI,
`test/input/images/tissue-mask` with one tissue mask, and `test/model` with model
weights file named `yolo11m_basicBox_img512.pt` (can be changed in code).
