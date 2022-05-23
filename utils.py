import logging
import os


def create_dir_if_not_exists(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_logger(module_name: str, model_name: str) -> logging.Logger:
    path = os.path.join("models", model_name, "logs", "log.log")
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s\n%(message)s")

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
