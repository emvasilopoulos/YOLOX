import torch
import exps.default.yolox_m
from yolox.utils import replace_module
from yolox.models.network_blocks import SiLU

EXPORT_INPUT_SIZE = (640, 640)  # (height, width)


def main():
    exp = exps.default.yolox_m.Exp()
    ckpt_file = "weights/yolox_m.pth"
    model = exp.get_model()
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, torch.nn.SiLU, SiLU)
    model.head.decode_in_inference = False
    dummy_input = torch.randn(1, 3, EXPORT_INPUT_SIZE[0], EXPORT_INPUT_SIZE[1])
    outputs = model(dummy_input)
    predictions = outputs["predictions"]
    low_res_feats = outputs["low_res_feats"]
    mid_res_feats = outputs["mid_res_feats"]
    high_res_feats = outputs["high_res_feats"]
    """ ------------ ONNX Export ------------- """
    # input_names = ["input"]
    # output_names = ["output"]
    # onnx_file = "yolox_m.onnx"
    # torch.onnx._export(
    #     model,
    #     dummy_input,
    #     onnx_file,
    #     input_names=input_names,
    #     output_names=output_names,
    #     opset_version=11,
    # )
    """ ---------- CoreML ---------- """
    try:
        import coremltools as ct
    except ImportError:
        raise SystemExit("coremltools is required for CoreML export. "
                         "Install it with `pip install coremltools`.")

    traced_model = torch.jit.trace(model, dummy_input)
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
        compute_precision=ct.precision.FLOAT32)
    mlmodel.save(
        f"yolox_m_{EXPORT_INPUT_SIZE[0]}x{EXPORT_INPUT_SIZE[1]}.mlpackage")


if __name__ == "__main__":
    main()
