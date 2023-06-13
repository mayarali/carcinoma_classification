import numpy as np

from agent.General_Agent import General_Agent
import json
from types import SimpleNamespace


def mean_results(aggregate_results):
    total_results = {}
    for img in aggregate_results[0]:
        total_results[img] = np.zeros(3)
        for cls in aggregate_results:
            if (aggregate_results[cls][img] < 0 ).any():
                aggregate_results[cls][img] = aggregate_results[cls][img] + aggregate_results[cls][img].min()
            aggregate_results[cls][img] = aggregate_results[cls][img] / aggregate_results[cls][img].sum()
            total_results[img] += aggregate_results[cls][img]

    return total_results


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

        if config.dataset.data_split.split_method == "kfold":
            aggregate_results = {}
            for fold in range(config.dataset.data_split.split_fold_num):
                config.dataset.data_split.split_fold = fold

                agent = General_Agent(config=config)
                results = agent.run()
                aggregate_results[fold] = results

            final_results = mean_results(aggregate_results)
            agent.save_unlabelled(results=final_results)

main()