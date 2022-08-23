import os

import torch

from gigadetect.models.yolov5.yolo import Model
from gigadetect.utils.logger import DEFAULT_LOGGER

def build_model(arch_name, num_classes):
    if arch_name not in ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
        raise ValueError(f'Unsupported yolov5 architecture type: {arch_name}')
        # check model architect config path
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'yolov5', arch_name + '.yaml')
    if not os.path.exists(config_path):
        raise ValueError(f'Yolov5 structure config file not found: {config_path}')
    single_model = Model(config_path, nc=num_classes)
    return single_model


def load_pretrained_state_dict(model, state_dict, device):
    if device.type != 'cpu':
        model.cuda()

    exclude = ['anchor']  # exclude keys
    state_dict = {k: v for k, v in state_dict.items()
                     if k in model.state_dict() and not any(x in k for x in exclude)
                     and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(state_dict, strict=False)
    return model

# used for inference, only required parameter provided
def load_yolov5_model(pretrained_model_path, device, num_classes=None, arch=None):
    """
    Used in pure evaluation procedure.
    We need to build model structure from the `model_path`, and build the model, and then load the state_dict()

    :param pretrained_model_path:
    :param device:
    :param num_classes:
    :param arch:
    :return:
    """

    # check model_path valid
    DEFAULT_LOGGER.info(f'model_path file is {pretrained_model_path}')
    if not os.path.exists(pretrained_model_path):
        raise ValueError(f'weights file not exist: {pretrained_model_path}')

    # get arch_name and state_dict from pretrained_model
    model_dict = torch.load(pretrained_model_path, map_location=device)
    if arch is None and 'arch' not in model_dict:
        msg = f'`arch` key does not exist in model dict, please check the version of {pretrained_model_path}.'
        DEFAULT_LOGGER.error(msg)
        raise ValueError(msg)
    if 'arch' not in model_dict:
       model_dict['arch'] = arch
    arch_name = model_dict['arch']
    DEFAULT_LOGGER.info(f'[Evaluate] Current model architecture name is: {arch_name}')


    if num_classes is None and 'num_classes' not in model_dict:
        msg = f'`num_classes` key does not exist in model dict, please check the version of {pretrained_model_path}.'
        DEFAULT_LOGGER.error(msg)
        raise ValueError(msg)
    if 'num_classes' not in model_dict:
        model_dict['num_classes'] = num_classes
    num_classes = model_dict['num_classes']
    DEFAULT_LOGGER.info(f'[Evaluate] Current model num_classes is: {num_classes}')


    state_dict = model_dict['state_dict']
    # state_dict = model_dict

    # build model
    single_model = build_model(arch_name, num_classes)

    # load pretrained parameters
    single_model = load_pretrained_state_dict(single_model, state_dict, device)

    # return
    return single_model


# used for evaluate, provided detailed cfg info
def load_pretrained_model(cfg, device):

    # 01. must have pretrained path
    pretrained_model_path = cfg.MODEL.WEIGHTS
    DEFAULT_LOGGER.info(f'model_path file is {pretrained_model_path}')
    if not os.path.exists(pretrained_model_path):
        raise ValueError(f'weights file not exist: {pretrained_model_path}')

    # 02. get empty model
    single_model = initialize_yolov5_model(cfg.MODEL.CFG_FILE_PATH, cfg.DATASETS.NUM_CLASSES, device)

    # 03. load arch_name and state_dict from pretrained_model
    model_dict = torch.load(pretrained_model_path, map_location=device)
    if 'arch' in model_dict:
        arch_name = model_dict['arch']
        DEFAULT_LOGGER.info(f'[Evaluate] Loaded pretrained model architecture name is: {arch_name}')

    # 04. load state_dict
    state_dict = model_dict['state_dict']
    single_model = load_pretrained_state_dict(single_model, state_dict, device)
    return single_model

def initialize_yolov5_model(arch_config_file_path, num_classes, device):
    model = Model(arch_config_file_path, nc=num_classes).to(device)
    return model