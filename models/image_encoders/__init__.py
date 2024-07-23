from typing import Tuple

from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder
from models.image_encoders.resnet import ResNet50Layer4Lower, ResNet50Layer4Upper
from models.image_encoders.modified_resnet import ModifiedResNetLower, ModifiedResNetUpper


def image_encoder_factory(config: dict) -> Tuple[AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder]:
    model_code = config['image_encoder']
    feature_size = config['feature_size']
    pretrained = config.get('pretrained', True)
    norm_scale = config.get('norm_scale', 4)
    checkpoint_path = config.get('ckpt_path', None)  # 获取预训练权重的路径

    if model_code == 'resnet50_layer4':
        lower_encoder = ResNet50Layer4Lower(pretrained, stride=config['stride'])
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = ResNet50Layer4Upper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
    elif model_code == 'modified_resnet':
        lower_encoder = ModifiedResNetLower(pretrained, layers=[3, 4, 6, 3], output_dim=512, heads=8,
                                            input_resolution=(224, 224), width=64, checkpoint_path=checkpoint_path)
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = ModifiedResNetUpper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
    else:
        raise ValueError("There's no image encoder matched with {}".format(model_code))

    return lower_encoder, upper_encoder