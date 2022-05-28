import logging

import hydra
import t5
import tensorflow.compat.v1 as tf
from omegaconf import DictConfig
import t5.models
from task_registry import register_datasets, DEFAULT_VOCAB

from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

TPU_TOPOLOGY = "v3-8"


def configure_runtime(tpu_name):
    tf_log = tf.get_logger()
    tf_log.removeHandler(tf_log.handlers[0])

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tpu_address = tpu.get_master()
        logging.root.info("Running on TPU: %s", tpu_address)
    except ValueError:
        raise Exception(f"Failed to connect to {tpu_name} TPU")

    tf.enable_eager_execution()
    tf.config.experimental_connect_to_host(tpu_address)
    tf.disable_v2_behavior()

    return tpu_address


def run_training(config):
    model = t5.models.MtfModel(
        model_dir=config.model_dir,
        # tpu=tpu_address,
        tpu_topology=TPU_TOPOLOGY,
        **config.model,
    )

    org_ckpt = t5.models.utils.get_latest_checkpoint_from_dir(config.pretrained_dir)
    try:
        last_ckpt = t5.models.utils.get_latest_checkpoint_from_dir(config.model_dir)
    except ValueError:
        last_ckpt = org_ckpt

    if last_ckpt < org_ckpt + config.finetune_steps:
        model.finetune(
            mixture_or_task_name=config.train_task,
            pretrained_model_dir=config.pretrained_dir,
            finetune_steps=config.finetune_steps,
        )
    else:
        logger.info("Finetuning already completed, skipping")

    for predict_file in config.predict_files:
        logger.info("Predicting file %s", predict_file)

        ts = datetime.utcnow().strftime("%Y%m%d")
        suffix = config.model_dir.strip("/").split("/")[-1]
        outfile = predict_file + f"-{suffix}-{ts}"
        logger.info("Writing to %s", outfile)

        model.predict(
            predict_file,
            outfile,
            vocabulary=DEFAULT_VOCAB,
            checkpoint_steps="all",
        )


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # tpu_address = configure_runtime(cfg.tpu_name)
    register_datasets(cfg.datasets)
    run_training(cfg)


if __name__ == "__main__":
    main()
