import logging

import hydra
import t5
import tensorflow.compat.v1 as tf
from omegaconf import DictConfig
import t5.models
from task_registry import register_datasets, get_vocab
import torch
import functools
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP

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


def run_tf_training(config):
    model = t5.models.MtfModel(
        model_dir=config.model_dir,
        tpu=None,
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
            vocabulary=get_vocab(config.vocab),
            checkpoint_steps="all",
        )


def run_torch_training(config):
    model = t5.models.HfPyTorchModel(
        model_spec=config.model.model_spec,
        # model_spec="/home/m.pioro/nlp/project/poleval-2021/pytorch_model.bin",
        model_dir="./pytorch",
        device=torch.device(config.model.device),
        # device=torch.device("cpu"),
    )
    # print(config.model)
    # exit()
    # if config.finetune_steps > 0:
    #     model.train(
    #         mixture_or_task_name=config.train_task,
    #         steps=config.finetune_steps,
    #         save_steps=config.model.save_checkpoints_steps,
    #         sequence_length={
    #             "inputs": config.model.sequence_length.inputs,
    #             "targets": config.model.sequence_length.targets,
    #         },
    #         split="train",
    #         batch_size=config.model.batch_size,
    #         optimizer=functools.partial(
    #             transformers.AdamW, lr=config.model.learning_rate_schedule
    #         ),
    #     )
    model_dict = {
        k[7:]: v
        for k, v in torch.load(
            "/home/m.pioro/nlp/project/poleval-2021/outputs/2022-06-23/00-29-06/pytorch/model-20000.checkpoint",
            # map_location=torch.device("cpu"),
        ).items()
    }
    model._model.load_state_dict(model_dict)
    if config.model.data_parallel:
        print("parallelized")
        model._model = torch.nn.parallel.DataParallel(model._model)
    model._model.eval()
    for predict_file in config.predict_files:
        logger.info("Predicting file %s", predict_file)
        ts = datetime.utcnow().strftime("%Y%m%d")
        suffix = config.model_dir.strip("/").split("/")[-1]
        outfile = predict_file + f"-{suffix}-{ts}"
        logger.info("Writing to %s", outfile)
        model.predict(
            inputs=predict_file,
            output_file=outfile,
            sequence_length=512,
            batch_size=16,
            vocabulary=get_vocab(config.spm_path),
            checkpoint_steps=...,
        )


def run_training(config):
    if hasattr(config.model, "pytorch_model") and getattr(
        config.model, "pytorch_model"
    ):
        run_torch_training(config)
    else:
        run_tf_training(config)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    if "spm_path" not in cfg:
        cfg["spm_path"] = "../../../sentencepiece.model"
    register_datasets(cfg.datasets, cfg.spm_path)
    run_training(cfg)


if __name__ == "__main__":
    main()
