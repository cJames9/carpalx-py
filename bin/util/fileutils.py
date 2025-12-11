import configparser
import os


def resolve_path(carpalx_dir, file):
    if os.path.isabs(file):
        return file
    else:
        return os.path.normpath(os.path.join(carpalx_dir, file))


def parse_conf_file(file, filepath='./etc'):
    if os.path.isfile(file):
        file = file
    else:
        file = os.path.join(filepath, file)
    if not os.path.exists(file):
        raise FileNotFoundError(f'cannot find or read configuration file {file}')

    conf_file = configparser.ConfigParser()
    conf_file.read(file)

    # TODO: used for debugging, remove later
    # for section in conf_file.sections():
    #     for option in conf_file[section]:
    #         if conf_file.get(section, option).endswith('.conf'):
    #             nested_file = conf_file.get(section, option)
    #             print(f'make sure to import {nested_file} related to {file}')

    return conf_file