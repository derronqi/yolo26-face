#!/usr/bin/env python3
"""
Export YOLO-Pose model with 9 separate outputs (box, cls, kpt for each stride)
Directly uses torch.onnx.export to preserve our custom forward logic
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path


def export_pose_separate(model_path, output_name=None, imgsz=[288, 512], opset=11):
    """Export Pose model with 9 separate outputs"""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get the pose model
    if hasattr(model.model, 'model'):
        pose_model = model.model
    else:
        pose_model = model
    
    # Set export mode to enable our custom forward logic
    head = pose_model.model[-1]
    head.export = True
    
    print(f"✓ Head type: {head.__class__.__name__}")
    print(f"✓ Export mode: ON (9 separate outputs)")
    
    # Wrap model to ensure proper forward
    class ExportWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            return self.model(x)
    
    wrapper = ExportWrapper(pose_model)
    wrapper.eval()
    
    # Test forward
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1])
    
    with torch.no_grad():
        outputs = wrapper(dummy_input)
    
    if isinstance(outputs, (list, tuple)):
        print(f"✓ Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            if isinstance(out, torch.Tensor):
                print(f"  Output {i}: {out.shape}")
    else:
        print(f"✗ Unexpected output type: {type(outputs)}")
        return
    
    # Generate output name
    weight_path = Path(model_path)
    if output_name is None:
        output_name = str(weight_path.parent / f"{weight_path.stem}_separate.onnx")
    
    # Define output names
    output_names = [
        'stride8_box', 'stride8_cls', 'stride8_kpt',
        'stride16_box', 'stride16_cls', 'stride16_kpt',
        'stride32_box', 'stride32_cls', 'stride32_kpt'
    ]
    
    # Export to ONNX
    print(f"\nExporting to {output_name}...")
    print(f"Output names: {output_names}")
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_name,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=output_names
    )
    
    print(f"\n✅ Export success: {output_name}")
    print(f"\nOutput format (9 outputs, NCHW):")
    print(f"  Stride 8:")
    print(f"    - stride8_box:  [batch, 4*reg_max, H8, W8]")
    print(f"    - stride8_cls:  [batch, nc, H8, W8]")
    print(f"    - stride8_kpt:  [batch, nk, H8, W8] (raw kpts + sigmoid visibility)")
    print(f"  Stride 16:")
    print(f"    - stride16_box: [batch, 4*reg_max, H16, W16]")
    print(f"    - stride16_cls: [batch, nc, H16, W16]")
    print(f"    - stride16_kpt: [batch, nk, H16, W16] (raw kpts + sigmoid visibility)")
    print(f"  Stride 32:")
    print(f"    - stride32_box: [batch, 4*reg_max, H32, W32]")
    print(f"    - stride32_cls: [batch, nc, H32, W32]")
    print(f"    - stride32_kpt: [batch, nk, H32, W32] (raw kpts + sigmoid visibility)")
    
    # Simplify if possible
    try:
        import onnxslim
        print(f"\nSimplifying ONNX model...")
        onnx_model = onnxslim.slim(output_name)
        onnx_model.save(str(output_name))
        print(f"✓ Model slimmed")
    except Exception as e:
        print(f"Warning: Could not slim: {e}")
    
    return output_name


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export YOLO-Pose with 9 separate outputs')
    parser.add_argument('--weights', type=str, default='runs/pose/train/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ONNX file name')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[256, 256],
                       help='Input image size [height, width]')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version')
    
    args = parser.parse_args()
    
    export_pose_separate(
        model_path=args.weights,
        output_name=args.output,
        imgsz=args.imgsz,
        opset=args.opset
    )
