# import models
from interpretors.gptoss import GPT_OSS_Model

interpretors = [GPT_OSS_Model]

def get_interpretor_class(interp_name):
    for cls in interpretors:
        if cls.__name__ == interp_name:
            return cls
    raise ValueError(f"{interp_name} not registered. Valid options are {[i.__name__ for i in interpretors]}")
