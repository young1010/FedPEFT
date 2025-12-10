#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import logging.config
import yaml
import glob
import json
from collections import OrderedDict

def load_config(config_dir, experiment_id):
    params = load_model_config(config_dir, experiment_id)
    data_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(data_params)
    return params

def load_model_config(config_dir, experiment_id):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if 'Base' in config_dict:
                found_params['Base'] = config_dict['Base']
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    # Update base and exp_id settings consectively to allow overwritting when conflicts exist
    params = found_params.get('Base', {})
    params.update(found_params.get(experiment_id, {}))
    assert "dataset_id" in params, f'expid={experiment_id} is not valid in config.'
    params["model_id"] = experiment_id
    return params

def load_dataset_config(config_dir, dataset_id):
    params = {"dataset_id": dataset_id}
    dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, "dataset_config/*.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f'dataset_id={dataset_id} is not found in config.')

def set_logger(params):
    dataset_id = params['dataset_id']
    model_id = params.get('model_id', '')
    log_dir = os.path.join(params.get('model_root', './checkpoints'), dataset_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, model_id + '.log')

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]: 
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'),
                                  logging.StreamHandler()])

def print_to_json(data, sort_keys=True):
    new_data = dict((k, str(v)) for k, v in data.items())
    if sort_keys:
        new_data = OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)

def print_to_list(data):
    return ' - '.join('{}: {:.6f}'.format(k, v) for k, v in data.items())

def calculate_model_size(param):
    if param.requires_grad:
        num_params = param.numel()
    else:
        num_params = 0
    param_size_in_bytes = param.element_size() * param.numel()
    model_size_in_mb = param_size_in_bytes / (1024 * 1024)
    return num_params, model_size_in_mb