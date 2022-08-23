import glob
import os
from pathlib import Path

import yaml


def search_file_under_directory(file):
    # Searches for file if not found locally
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]


def fetch_image_path_list(path, valid_image_format_list=[]):
    """
    given a `path`, return corresponding images' paths list
    `path` could be a `.txt` file, which contains the image_path line by line.
    `path` could be a folder, which contains all the images
    :param path:
    :param valid_image_format_list:
    :return:
    """
    try:
        path = str(Path(path))  # os-agnostic
        parent = str(Path(path).parent) + os.sep
        if os.path.isfile(path):  # file
            with open(path, 'r') as f:
                f = f.read().splitlines()
                f = [x.replace('./', parent) if x.startswith('./') else x for x in f]  # local to global path
        elif os.path.isdir(path):  # folder
            f = glob.iglob(path + os.sep + '*.*')
        else:
            raise Exception('%s does not exist' % path)
        img_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in valid_image_format_list]
        return img_files
    except:
        raise Exception('Error loading data from %s' % path)


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def load_config_dict(yaml_file_path):
    yaml_file_path = search_file_under_directory(yaml_file_path)
    with open(yaml_file_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    return data