from torchvision import transforms

IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

''' 创建训练图像的变换 '''
def get_train_transform(transform_config: dict):
    # 从配置字典transform_config中读取use_transform和img_size两个键值
    use_transform = transform_config['use_transform']
    img_size = transform_config['img_size']

    # 如果use_transform为真（True），则使用一系列更复杂的变换
    if use_transform:
        return transforms.Compose([transforms.RandomResizedCrop(size=img_size, scale=(0.75, 1.33)), # 随机大小和比例裁剪图像
                                   transforms.RandomHorizontalFlip(), # 随机水平翻转图像
                                   transforms.ToTensor(), # 将图像转换为PyTorch张量
                                   transforms.Normalize(**IMAGENET_STATS)]) # 使用ImageNet的均值和标准差来标准化图像

    # 如果use_transform为假（False），则使用较简单的变换
    return transforms.Compose([transforms.Resize((img_size, img_size)), # 调整图像大小
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET_STATS)])

'''' 创建验证图像的变换 '''
def get_val_transform(transform_config: dict):
    # 只从配置字典中读取img_size键值
    img_size = transform_config['img_size']

    # 并应用以下变换
    return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET_STATS)])


def image_transform_factory(config: dict):
    return {
        'train': get_train_transform(config),
        'val': get_val_transform(config)
    }
