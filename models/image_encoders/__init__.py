from typing import Tuple

from trainers.abc import AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder
from models.image_encoders.resnet import ResNet18Layer4Lower, ResNet18Layer4Upper, ResNet50Layer4Lower, \
    ResNet50Layer4Upper,ResNet50x4Lower, ResNet50x4Upper
from models.image_encoders.modified_resnet import ModifiedResNetLower, ModifiedResNetUpper
from models.image_encoders.vit import VisionTransformerLower, VisionTransformerUpper


def image_encoder_factory(config: dict) -> Tuple[AbstractBaseImageLowerEncoder, AbstractBaseImageUpperEncoder]:
    model_code = config['image_encoder']
    feature_size = config['feature_size']
    pretrained = config.get('pretrained', True)
    norm_scale = config.get('norm_scale', 4)

    if model_code == 'resnet18_layer4':
        lower_encoder = ResNet18Layer4Lower(pretrained)
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = ResNet18Layer4Upper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
    elif model_code == 'resnet50_layer4':
        lower_encoder = ResNet50Layer4Lower(pretrained, stride=config['stride'])
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = ResNet50Layer4Upper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
    elif model_code == 'resnet50x4_layer4':
        lower_encoder = ResNet50x4Lower(pretrained, stride=config.get('stride', False))
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = ResNet50x4Upper(lower_feature_shape, feature_size, pretrained=pretrained,
                                        norm_scale=norm_scale)
    elif model_code == 'modified_resnet':
        lower_encoder = ModifiedResNetLower(pretrained)
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = ModifiedResNetUpper(lower_feature_shape, feature_size, pretrained=pretrained,
                                            norm_scale=norm_scale)
    elif model_code == 'vision_transformer':
        input_resolution = (224,224)
        patch_size = config['patch_size']
        stride_size = config['stride_size']
        width = config['width']
        layers = config['layers']
        heads = config['heads']
        output_dim = config['output_dim']
        lower_encoder = VisionTransformerLower(input_resolution, patch_size, stride_size, width, layers, heads,
                                               output_dim)
        lower_feature_shape = lower_encoder.layer_shapes()['layer4']
        upper_encoder = VisionTransformerUpper(lower_feature_shape, feature_size, pretrained=pretrained,
                                               norm_scale=norm_scale)
    else:
        raise ValueError("There's no image encoder matched with {}".format(model_code))

    return lower_encoder, upper_encoder