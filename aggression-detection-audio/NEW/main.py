#!/usr/bin/env python3
import argparse, json, os
import torch

from utils.Logger import Logger
from test.AutoencoderEvaluator import AutoencoderEvaluator

import model as mdl

from train.Trainer import Trainer
from torchsummary import summary

import manager.SoundManager as manager
import transforms as transforms


def _get_transform(config):
    tsf_mode = config["data"]["format"]
    tsf_name = config["transforms"]["type"]
    tsf_args = config["transforms"]["args"]
    return getattr(transforms, tsf_name)(tsf_args)

def _get_model_att(checkpoint):
    m_name = checkpoint["config"]["model"]["type"]
    sd = checkpoint["state_dict"]
    return m_name, sd

def eval_main(checkpoint):
    config = checkpoint["config"]
    data_config = config["data"]

    tsf = _get_transform(config)

    soundManager = getattr(manager, config["data"]["type"])(config["data"])
    testLoader = soundManager.get_testing_loader(tsf)

    m_name, sd = _get_model_att(checkpoint)
    model = getattr(mdl, m_name)(config, state_dict=sd)

    model.load_state_dict(checkpoint["state_dict"])

    evaluation = AutoencoderEvaluator(testLoader, model)
    result = evaluation.evaluate()

    return result

def train_main(config, resume):
    train_logger = Logger()
    data_config = config["data"]

    t_transforms = _get_transform(config)
    v_transforms = _get_transform(config)
    print('*' * 50)
    print('*' * 50)
    print(t_transforms)

    soundManager = getattr(manager, config["data"]["type"])(config["data"])

    t_loader = soundManager.get_training_loader("train", t_transforms)
    v_loader = soundManager.get_training_loader("test", v_transforms)
    m_name = config["model"]["type"]
    model = getattr(mdl, m_name)(config=config)

    print('*' * 50)
    print('*' * 50)
    ae = mdl.AutoEncoderConv()
    ae = ae.to("cuda")
    print(summary(ae, (1, 128, 87)))

    loss = getattr(mdl, config["train"]["loss"])

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    opt_name = config["optimizer"]["type"]
    opt_args = config["optimizer"]["args"]
    optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)

    lr_name = config["lr_scheduler"]["type"]
    lr_args = config["lr_scheduler"]["args"]
    if lr_name == "None":
        lr_scheduler = None
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

    trainer = Trainer(model, loss, optimizer,
                      resume=resume,
                      config=config,
                      data_loader=t_loader,
                      valid_data_loader=v_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()
    return trainer

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="PyTorch Template")

    argparser.add_argument("action", type=str,
                           help="what action to take (train, test, eval)")
    argparser.add_argument("-c", "--config", default=None, type=str,
                           help="config file path (default: None)")
    argparser.add_argument("-r", "--resume", default=None, type=str,
                           help="path to latest checkpoint (default: None)")
    argparser.add_argument("--model_mode", default="init", type=str,
                           help="type of transfer learning to use")

    args = argparser.parse_args()

    checkpoint = None
    if args.config:
        config = json.load(open(args.config))
        config["model_mode"] = args.model_mode
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        config = checkpoint["config"]
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.action == "train":
        train_main(config, args.resume)

    elif args.action == "eval":
        eval_main(checkpoint)