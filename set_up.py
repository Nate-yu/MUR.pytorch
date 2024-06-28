import json
import os
import pprint as pp
import random
from datetime import date

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

''' 固定随机种子，确保实验可复现 '''
def fix_random_seed_as(random_seed):
    # 如果 random_seed 为 -1，则随机生成一个种子，并打印出来
    if random_seed == -1:
        random_seed = np.random.randint(100000)
        print("RANDOM SEED: {}".format(random_seed))

    # 设置 Python 的内置 random 模块、numpy、torch（包括 CUDA 随机种子）以及 cudnn 的随机种子和行为，确保实验结果的一致性
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    return random_seed


def _get_experiment_index(experiment_path):
    idx = 0
    while os.path.exists(experiment_path + "_" + str(idx)):
        idx += 1
    return idx

''' 创建实验导出文件夹 '''
def create_experiment_export_folder(experiment_dir, experiment_description):
    # 根据实验目录和描述创建一个新的文件夹用于存放实验结果
    print(os.path.abspath(experiment_dir))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    experiment_path = get_name_of_experiment_path(experiment_dir, experiment_description)
    # 打印并创建实验目录和路径，确保实验的输出有明确的存储位置
    print(os.path.abspath(experiment_path))
    os.mkdir(experiment_path)
    print("folder created: " + os.path.abspath(experiment_path))
    return experiment_path


def get_name_of_experiment_path(experiment_dir, experiment_description):
    experiment_path = os.path.join(experiment_dir, (experiment_description + "_" + str(date.today())))
    idx = _get_experiment_index(experiment_path)
    experiment_path = experiment_path + "_" + str(idx)
    return experiment_path

''' 导出配置文件为 JSON '''
def export_config_as_json(config, experiment_path):
    # 将实验的配置信息导出到 JSON 文件中，保存在实验目录下
    with open(os.path.join(experiment_path, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=2)

''' 生成和设置标签 '''
def generate_tags(config):
    tags = []
    tags.append(config.get('generator', config.get('text_encoder')))
    tags.append(config.get('trainer'))
    tags = [tag for tag in tags if tag is not None]
    return tags

''' 设置 GPU '''
def set_up_gpu(device_idx):
    # 根据传入的 device_idx（设备索引）设置环境变量 CUDA_VISIBLE_DEVICES，以指定哪些 GPU 可用
    if device_idx: # 如果 device_idx 有值，则直接使用该值；如果没有值，则使用环境变量中已设置的值
        os.environ['CUDA_VISIBLE_DEVICES'] = device_idx
        # 函数返回 GPU 的数量
        return {
            'num_gpu': len(device_idx.split(","))
        }
    else:
        idxs = os.environ['CUDA_VISIBLE_DEVICES']
        return {
            'num_gpu': len(idxs.split(","))
        }

''' 配置和初始化实验环境 '''
def setup_experiment(config):
    # 设置 GPU
    device_info = set_up_gpu(config['device_idx'])
    config.update(device_info) # 将gpu信息更新到配置字典 config 中

    # 固定随机种子
    random_seed = fix_random_seed_as(config['random_seed'])
    config['random_seed'] = random_seed

    # 创建实验导出文件夹
    export_root = create_experiment_export_folder(config['experiment_dir'], config['experiment_description'])
    # 导出配置文件为 JSON
    export_config_as_json(config, export_root)
    config['export_root'] = export_root

    # 打印配置信息
    pp.pprint(config, width=1)
    # 设置 WANDB 环境变量
    os.environ['WANDB_SILENT'] = "true"
    # 生成和设置标签
    tags = generate_tags(config)
    project_name = config['wandb_project_name']
    wandb_account_name = config['wandb_account_name']
    experiment_name = config['experiment_description']
    experiment_name = experiment_name if config['random_seed'] != -1 else experiment_name + "_{}".format(random_seed)
    # 初始化 Weights & Biases 实验
    wandb.init(config=config, name=experiment_name, project=project_name,
               entity=wandb_account_name, tags=tags)
    # 返回 export_root（实验的根目录路径）和更新后的 config 配置字典
    return export_root, config
