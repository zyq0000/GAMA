import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.__dict__.update(config)

        for k, v in args.__dict__.items():
            if k not in self.__dict__:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())
