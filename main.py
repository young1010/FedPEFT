#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
from datetime import datetime
from framework.utils import load_config, set_logger, print_to_json, print_to_list
from framework.modules.utils import seed_everything
import zoo as model_zoo
import dataloaders as dataload_zoo
import gc
import argparse
import os
from pathlib import Path

if __name__ == '__main__':
    ''' Usage: python main.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['device'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    dataload_class = getattr(dataload_zoo, params['dataloader'])
    dataload = dataload_class(**params)

    model_class = getattr(model_zoo, params['model'])
    model = model_class(dataload=dataload, **params)
    # model.count_parameters() # print number of parameters used in model

    test_results = model.fit()

    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.",  print_to_list(test_results)))