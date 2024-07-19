from language.vocabulary import AbstractBaseVocabulary
from models.text_encoders.clip import CLIPTextEncoder
from trainers.abc import AbstractBaseTextEncoder
from models.text_encoders.roberta import RobertaEncoder, BertFc


def text_encoder_factory(vocabulary: AbstractBaseVocabulary, config: dict) -> AbstractBaseTextEncoder:
    model_code = config['text_encoder']  # 'roberta' 或 'clip'
    feature_size = config['text_feature_size'] # 512

    if model_code == RobertaEncoder.code():
        return RobertaEncoder(feature_size=feature_size), BertFc(feature_size=feature_size)
    elif model_code == CLIPTextEncoder.code():
        return CLIPTextEncoder(config=config), None
    else:
        raise ValueError("There's no text encoder matched with {}".format(model_code))
