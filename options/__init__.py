from options.command_line import load_config_from_command
from options.config_file import load_config_from_file

''' 合并配置：将命令行配置、文件配置和用户定义的配置合并成一个字典。
    这个方法按照提供的列表顺序（优先级递增）合并配置，后面的配置将覆盖前面的配置 '''
def _merge_configs(configs_ordered_by_increasing_priority):
    merged_config = {}
    for config in configs_ordered_by_increasing_priority:
        for k, v in config.items():
            merged_config[k] = v
    return merged_config

''' 检查必需的配置：检查从配置文件中加载的配置。它确保所有必需的配置项都已定义，如果某些必需的配置项未在命令行或配置文件中定义，则会触发异常 '''
def _check_mandatory_config(config_from_config_file, user_defined_configs,
                            exception_keys=('experiment_description', 'device_idx')):
    exception_keys = [] if exception_keys is None else exception_keys
    trigger = False
    undefined_configs = []
    for key, val in config_from_config_file.items():
        if val == "":
            if key not in user_defined_configs and key not in exception_keys:
                trigger = True
                undefined_configs.append(key)
                print("Must define {} setting from command".format(key))
    if trigger:
        raise Exception('Mandatory configs not defined:', undefined_configs)

''' 根据合并后的配置生成实验描述 '''
def _generate_experiment_description(configs, config_from_command):
    experiment_description = configs['experiment_description']
    # 如果 experiment_description 为空，则从命令行配置中移除一些关键字，并使用剩余的配置项生成描述字符串
    if experiment_description == "":
        remove_keys = ['dataset', 'trainer', 'config_path', 'device_idx']
        for key in remove_keys:
            if key in config_from_command:
                config_from_command.pop(key)

        descriptors = []
        for key, val in config_from_command.items():
            descriptors.append(key + str(val))
        experiment_description = "_".join([configs['dataset'], configs['trainer'], *descriptors])
    return experiment_description


''' 整合来自命令行和配置文件的配置信息，并生成一个包含实验描述的最终配置字典 '''
def get_experiment_config():
    # 加载命令行配置
    config_from_command, user_defined_configs = load_config_from_command()
    # 加载文件配置
    config_from_config_file = load_config_from_file(config_from_command['config_path'])
    # 检查必需的配置
    _check_mandatory_config(config_from_config_file, user_defined_configs)
    # 合并配置
    merged_configs = _merge_configs([config_from_command, config_from_config_file, user_defined_configs])
    # 生成实验描述
    experiment_description = _generate_experiment_description(merged_configs, user_defined_configs)
    # 将生成的实验描述添加到合并后的配置字典中
    merged_configs['experiment_description'] = experiment_description
    # 返回这个包含完整配置和实验描述的字典
    return merged_configs
