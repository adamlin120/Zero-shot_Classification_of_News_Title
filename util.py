import hashlib


def get_config_sha1(config, digit):
    s = hashlib.sha1()
    s.update(str(config).encode('utf-8'))
    return s.hexdigest()[:digit]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)