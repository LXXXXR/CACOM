from collections import defaultdict
import logging
import numpy as np

import wandb
class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def log_stat(self, key, value, t):
        self.stats[key].append((t, value))

        wandb.log({
            key: value,
            "timestep": t
        })
        
    def log_histogram(self, key, value, t):
        pass

    def log_embedding(self, key, value):
        pass

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            import torch as th
            item = "{:.4f}".format(th.mean(th.tensor([x[1] for x in self.stats[k][-window:]])))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name=name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        handlers.append(file_handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger