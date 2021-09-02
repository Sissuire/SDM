from __future__ import absolute_import, division, print_function

import os
import yaml
import sys

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), 'default_config.yaml')


def config_parser(config_file=None):
    if config_file is None:
        config_file = DEFAULT_CONFIG
        print('\nusing DEFAULT configuratioin.')
    else:
        print('\nConfig: %s' % config_file)

    # load default config data
    exist_config = os.path.isfile(config_file)

    if exist_config:
        if sys.version_info > (3, 0):
            # if Python 3
            with open(config_file, 'r', encoding='utf-8') as stream:
                try:
                    d_config_data = yaml.load(stream, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    print(exc)
        else:
            # if Python 2
            with open(config_file, 'r') as stream:
                try:
                    d_config_data = yaml.load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        db_config = d_config_data['database']
        model_config = d_config_data['model']
    else:
        print(' @ Config file does not exist: %s' % config_file)
        db_config = {}
        model_config = {}

    # # load the current config data
    # with open(config_file, 'r') as stream:
    #     try:
    #         config_data = yaml.load(stream)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #
    # use_common_config = False
    # if 'common' in list(config_data.keys()):
    #     base_config_data = config_data['common']
    #     use_common_config = True
    #
    # if section is not None:
    #     if section in list(config_data.keys()):
    #         config_data = config_data[section]
    #     else:
    #         raise ValueError('No %s in %s' % (section, config_file))
    # else:
    #     config_data = config_data
    #
    # # merge the current config into base config
    # if use_common_config:
    #     if 'database' in list(base_config_data.keys()):
    #         overwrite_config(db_config, base_config_data['database'])
    #     if 'model' in list(base_config_data.keys()):
    #         overwrite_config(model_config, base_config_data['model'])
    #     if 'training' in list(base_config_data.keys()):
    #         overwrite_config(train_config, base_config_data['training'])
    #
    # if 'database' in list(config_data.keys()):
    #     overwrite_config(db_config, config_data['database'])
    # if 'model' in list(config_data.keys()):
    #     overwrite_config(model_config, config_data['model'])
    # if 'training' in list(config_data.keys()):
    #     overwrite_config(train_config, config_data['training'])
    #
    # check_subsection(db_config)

    # if db_config['num_subsection']:
    #     if db_config['train']['num_subsection']:
    #         copy_config(db_config['train'], db_config['train'][0])
    #         copy_config(db_config, db_config['train'][0])
    #     else:
    #         copy_config(db_config, db_config['train'])

    show_configs(db_config, model_config)

    return db_config, model_config


def dump_config(filename, db_config, model_config, train_config):
    cfg = {}
    cfg['database'] = db_config.copy()
    cfg['model'] = model_config.copy()
    cfg['training'] = train_config.copy()
    # check_child_list(cfg)
    with open(filename, 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)


def check_child_list(parent_section):
    subsections = []
    # check if parent_section has subsections
    for key, value in parent_section.items():
        if isinstance(value, dict):
            subsections.append(key)

    if len(subsections) > 0:
        for subsection in subsections:
            # overwrite the copied child_section with new information
            check_child_list(parent_section[subsection])

    for key, value in parent_section.items():
        if isinstance(value, list):
            parent_section[key] = str(value)


def check_subsection(parent_section):
    subsections = []
    # check if parent_section has subsections
    for key, value in parent_section.items():
        if isinstance(value, dict):
            subsections.append(key)

    if len(subsections) > 0:
        for subsection in subsections:
            # copy parent_section to child_section
            child_section = {}
            for key, value in parent_section.items():
                if not isinstance(value, dict):
                    child_section[key] = value

            # overwrite the copied child_section with new information
            overwrite_config(child_section, parent_section[subsection])

            check_subsection(child_section)
            parent_section[subsection] = child_section

        # remove keys in parent_section
        for key in list(parent_section):
            if key not in subsections:
                parent_section.pop(key)

        parent_section['num_subsection'] = len(subsections)
    else:
        parent_section['num_subsection'] = 0


def overwrite_config(base_config, new_config):
    for key, value in new_config.items():
        base_config[key] = value


def copy_config(base_config, new_config):
    for key, value in new_config.items():
        if key not in base_config:
            base_config[key] = value


def show_configs(db_config, model_config):
    # if 'train' in db_config:
    #     print('Train Dataset: %s' % db_config['train']['sel_data'])
    #     print(' - Scenes:', db_config['train']['scenes'], end='')
    #     print(' / dist_types:', db_config['train']['dist_types'])
    #     print(' - Patch size:', db_config['train']['patch_size'], end='')
    #     print(' / Patch step:', db_config['train']['patch_step'])

    #     print('Test Dataset: %s' % db_config['test']['sel_data'])
    #     print(' - Scenes:', db_config['test']['scenes'], end='')
    #     print(' / dist_types:', db_config['test']['dist_types'])
    #     print(' - Patch size:', db_config['test']['patch_size'], end='')
    #     print(' / Patch step:', db_config['test']['patch_step'])
    # else:
    #     print('Dataset: %s' % db_config['sel_data'])
    #     print(' - Scenes:', db_config['scenes'], end='')
    #     print(' / dist_types:', db_config['dist_types'])
    #     print(' - Patch size:', db_config['patch_size'], end='')
    #     print(' / Patch step:', db_config['patch_step'])

    print('--' * 12)
    for key in db_config.keys():
        print('%s:  %s' % (key, db_config[key]))
    print('--' * 12)
    for key in model_config.keys():
        print('%s:  %s' % (key, model_config[key]))
    print('--' * 12)

