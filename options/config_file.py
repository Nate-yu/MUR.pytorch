import json

''' 加载文件配置 '''
def load_config_from_file(json_path):
    # 使用命令行配置中的 config_path 键来获取配置文件的路径
    if not json_path:
        return {}

    with open(json_path, 'r') as f:
        config = json.load(f)

    print("Config at '{}' has been loaded".format(json_path))
    # load_config_from_file() 方法使用此路径打开并读取 JSON 配置文件，返回一个字典 config_from_config_file，包含文件中的配置
    return config
