# ==========================================
# File: test.py
# Purpose: Evaluate trained ReID models on benchmark datasets
# Modified by: Hang Zhang (hzha0521)
# Date: 2025-09-15
# Based on: Original test.py from TransReID repo
# Notes:
#   - [NEW] Automatically switches to self-trained mode during evaluation
#           (PRETRAIN_CHOICE="self") when TEST.WEIGHT is provided.
#   - [NEW] Clears MODEL.PRETRAIN_PATH to prevent loading ImageNet weights.
#   - [NEW] Adds checkpoint existence check and clearer logging.
# ==========================================

import os
import sys
import argparse

from config import cfg
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Testing / Inference")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # --- Load and freeze config ---
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # --- Ensure output dir exists ---
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            logger.info("\n" + cf.read())
    logger.info("Running with config:\n{}".format(cfg))

    # --- Device env (cast to str; avoid tuple-like values) ---
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)

    # ==========================================================
    # === [Modified by: Hang Zhang (hzha0521), 2025-09-15] ===
    # === Modification content: 
    # - When cfg.TEST.WEIGHT is provided, force evaluation mode to
    #   PRETRAIN_CHOICE="self" and clear PRETRAIN_PATH (avoid reloading ImageNet weights)
    # - Add checkpoint existence check and clearer logging
    # ==========================================================
    if getattr(cfg.TEST, "WEIGHT", ""):
        test_weight = cfg.TEST.WEIGHT
        if not os.path.isfile(test_weight):
            logger.error(f"[ERROR] TEST.WEIGHT not found: {test_weight}")
            sys.exit(1)

        cfg.defrost()
        cfg.MODEL.PRETRAIN_CHOICE = "self"  # only load from TEST.WEIGHT
        cfg.MODEL.PRETRAIN_PATH = ""        # no external pretrained needed
        cfg.freeze()
        logger.info(f"[Eval] Using self-trained checkpoint: {test_weight}")
    else:
        logger.warning(
            "[WARN] TEST.WEIGHT is empty. You are evaluating without a trained checkpoint.\n"
            "       If this is unintended, pass --opts TEST.WEIGHT /path/to/ckpt.pth"
        )

    # --- Build data loaders ---
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # --- Build model ---
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # --- Load evaluation checkpoint (if provided) ---
    if getattr(cfg.TEST, "WEIGHT", ""):
        model.load_param(cfg.TEST.WEIGHT)

    # --- Inference ---
    if cfg.DATASETS.NAMES == "VehicleID":
        # VehicleID protocol: 10 trials
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5 = do_inference(cfg, model, val_loader, num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}".format(all_rank_1.sum() / 10.0, all_rank_5.sum() / 10.0))
    else:
        do_inference(cfg, model, val_loader, num_query)
