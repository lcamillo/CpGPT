"""Module for evaluating trained models using checkpoints and computing test metrics."""

from typing import TYPE_CHECKING, Any

import hydra
import rootutils
from omegaconf import DictConfig

from cpgpt.infer import CpGPTInferencer
from cpgpt.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

if TYPE_CHECKING:
    from lightning import LightningDataModule, LightningModule, Trainer
    from lightning.pytorch.loggers import Logger

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Tuple containing:
            - Dict[str, Any]: Dictionary with metrics
            - Dict[str, Any]: Dictionary with all instantiated objects

    Raises:
        ValueError: If trainer_ckpt_path is not specified in config

    """
    if not cfg.model_ckpt_path:
        msg = "model_ckpt_path must be specified in config"
        raise ValueError(msg)

    log.info(f"Using seed: 42")
    L.seed_everything(42, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    inferencer = CpGPTInferencer()
    config = inferencer.load_cpgpt_config(cfg.config_path)
    model = inferencer.load_cpgpt_model(config, cfg.model_ckpt_path, cfg.strict_load)

    if cfg.val:
        log.info("Starting validation!")
        trainer.validate(model=model, datamodule=datamodule)

    if cfg.test:
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
