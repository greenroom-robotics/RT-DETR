"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import datetime
from src.core import YAMLConfig
from src.solver._solver import BaseSolver
import json
import onnx
import onnxsim
import numpy as np
from onnx import TensorProto, helper


def add_meta(onnx_model, key, value):
    # Add meta to model
    meta = onnx_model.metadata_props.add()
    meta.key = key
    meta.value = value


def get_modality(config_path):
    name = os.path.splitext(os.path.basename(config_path))[0]
    tokens = name.split("_")
    if "rgb" in tokens:
        return "rgb"
    if "ir" in tokens:
        return "ir"
    return "unknown"


def main(
    args,
):
    """main"""
    resize_h, resize_w = (args.input_size, args.input_size)

    update_dict = {k: v for k, v in args.__dict__.items() if v is not None}
    update_dict["eval_spatial_size"] = [resize_h, resize_w]
    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # Export resolution can differ from checkpoint resolution, so ignore buffers that don't match.
        # Model will regenerate the buffers at desired export resolution.
        matched_state, infos = BaseSolver._matched_state(cfg.model.state_dict(), state)
        cfg.model.load_state_dict(matched_state, strict=False)
        print(f"Loaded model.state_dict, {infos}")

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()

    data = torch.rand(1, args.image_channels, resize_h, resize_w)
    size = torch.tensor([[resize_h, resize_w]])
    _ = model(data, size)

    # Enable dynamic batch size
    dynamic_axes = {
        "images": {
            0: "N",
        },
        "orig_target_sizes": {0: "N"},
    }

    torch.onnx.export(
        model,
        (data, size),
        args.output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        verbose=False,
        do_constant_folding=True,
    )
    classes = list(cfg.train_dataloader.dataset.category2name.values())

    onnx_model = onnx.load(args.output_file)
    add_meta(onnx_model, key="date", value=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    add_meta(onnx_model, key="classes", value=json.dumps(classes))
    add_meta(onnx_model, key="model", value="RT-DETR")
    add_meta(onnx_model, key="input_shape", value=json.dumps((resize_h, resize_w)))
    add_meta(onnx_model, key="modality", value=get_modality(args.config))
    onnx.save(onnx_model, args.output_file)

    if args.check:
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")

    if args.simplify:
        onnx_model_simplify, check = onnxsim.simplify(args.output_file)
        onnx.save(onnx_model_simplify, args.output_file)
        print(f"Successfully simplified onnx model: {check}...")

    if args.single_input:
        constant_value = np.array([[resize_h, resize_w]], dtype=np.int64)
        const_input(args.output_file, "orig_target_sizes", constant_value)


def const_input(model_path, input_name, constant_value):
    model = onnx.load(model_path)

    initializer = helper.make_tensor(
        name=input_name,
        data_type=TensorProto.INT64,
        dims=constant_value.shape,
        vals=constant_value.flatten(),
    )

    model.graph.initializer.append(initializer)

    inputs_to_keep = [inp for inp in model.graph.input if inp.name != input_name]

    del model.graph.input[:]
    model.graph.input.extend(inputs_to_keep)

    model_simplified, check = onnxsim.simplify(model)

    if check:
        onnx.save(model_simplified, model_path)
        print(f"Simplified model saved to {model_path}")
    else:
        print("Simplification failed, saving original modification")
        onnx.save(model, model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--resume", "-r", type=str)
    parser.add_argument("--output_file", "-o", type=str, default="model.onnx")
    parser.add_argument(
        "--input_size", "-s", type=int, default=1280, help="-s 640 for IR, -s 1280 for RGB"
    )
    parser.add_argument(
        "--image_channels",
        "-i",
        type=int,
        default=3,
        help="Number of image channels. IR has 1 channel",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--single_input",
        action="store_true",
        default=False,
        help="Add orig_target_sizes into a constant leaving `images` as the only input.",
    )
    args = parser.parse_args()

    main(args)
