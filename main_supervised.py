from agent.General_Agent import General_Agent
import json
from types import SimpleNamespace


def main():

    config_list = [
        "./configs/fully_supervised/simple_CNN.json"
        # "./configs/fully_supervised/simple_CNN_tiles.json"
    ]
    for conf in config_list:

        #Load config and turn it into class
        config = json.load(open(conf))
        config = json.dumps(config)
        config = json.loads(config, object_hook=lambda d: SimpleNamespace(**d))

        agent = General_Agent(config=config)
        agent.run()

main()