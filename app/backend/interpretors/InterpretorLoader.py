# import models
from interpretors.gptoss import GPT_OSS_Model
from interpretors.gptoss_tuned import GPT_OSS_Model_Tuned

interpretors = [GPT_OSS_Model, GPT_OSS_Model_Tuned]

def get_interpretor_class(interp_name):
    for cls in interpretors:
        if cls.__name__ == interp_name:
            return cls
    raise ValueError(f"{interp_name} not registered. Valid options are {[i.__name__ for i in interpretors]}")
