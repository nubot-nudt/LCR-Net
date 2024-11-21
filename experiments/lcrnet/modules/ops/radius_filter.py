import importlib


ext_module = importlib.import_module('utils.ext')


def radius_filter(nodes_dict, length_dict, radius):
    """radius_filter in stack mode.

    This function is implemented on CPU.

    Args:
        

    Returns:
        
    """
    masks, nms_length = ext_module.radius_filter(nodes_dict, length_dict, radius)
    return masks, nms_length
