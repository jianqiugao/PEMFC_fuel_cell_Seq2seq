import yaml
def _load_yml(path,):
    with open(path,encoding ='utf-8-sig') as f :
        conf = yaml.load(f,yaml.FullLoader)
    return conf

config =_load_yml('../config/config.yml')
