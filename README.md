A PyTorch project for GigaPixel image detection.

# GigaDet

GigaDet is a progressive object detection framework, which consists of a PGN module and a DecDet module.

In this repository, the features below are implemented:
- train for DecDet
- inference for DecDet
- inference for GigaDet (Depends on PGN project codes) 

## Usage

### DecDet
#### Train DecDet on COCO Dataset
```
PYTHONPATH=gigadetect:$PYTHONPATH python example/decdet_train.py --config-file configs/GigaDet/GigaDet_COCO_Tiny.yml
```

#### Evaluate DecDet on PANDA Test Set
```
 PYTHONPATH=gigadetect:$PYTHONPATH python example/decdet_evaluate.py --config-file configs/GigaDet/GigaDet_COCO.yml
```
Some key config items in config file:
- MODEL.WEIGHTS: Specifies the pre-trained model file.
- DATASETS.VAL_ROOR: Specifies the dataset which should be evaluated.
- MODEL.CFG_FILE_PATH: Specifies the config_file (such as `yolov5s.yaml`) for specific arch corresponding to the loaded model.


### GigaDet

#### GigaDetSingle Image Forward
```
PYTHONPATH=.:path/to/pgn/repo python example/detect_panda_image.py --image_path path/to/image --config-file path/to/config/file --save_path path/to/save/result
```

#### Evaluate the whole procedure
```
python example/evaluate_panda_detect_service.py --config-file path/to/config/file 
```




# Update

- 2020-08-11: Create the repository.
- 2021-02-02: Start to refactor the repository.
- 2021-03-04: continue to refactor.