from models.hierarchical import HGRU
from models.x_vector import XVector


def get_model(num_features, num_classes, model_tag):
    # Model should be an object of Base(models.base.Base) Class.
    if model_tag == 'HGRU':
        return HGRU(num_features, num_classes)
    elif model_tag == 'XVECTOR':
        return XVector(num_features, num_classes)
    else:
        raise Exception('Model {} not defined in input_model.py'.format(model_tag))
