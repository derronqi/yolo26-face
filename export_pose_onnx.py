import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Pose, Pose26
from pathlib import Path


class PoseSeparateExporter(nn.Module):
    """Wrapper to export YOLO26-Pose model with 9 separate outputs"""
    
    def __init__(self, model):
        super().__init__()
        # model is the full PoseModel
        self.model = model
        
    def forward(self, x):
        """
        Forward pass mimicking BaseModel._predict_once but with separate outputs
        """
        y = []  # outputs from each layer
        
        # Iterate through all layers
        for m in self.model.model:
            if m.f != -1:  # if not from previous layer
                # Get input from earlier layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            # Special handling for the last layer (head)
            if m == self.model.model[-1]:
                # This is the Pose/Pose26 head - return 9 separate outputs
                return self._forward_head(m, x)
            
            # Normal layer forward
            x = m(x)
            y.append(x if m.i in self.model.save else None)
        
        return x
    
    def _forward_head(self, head, x):
        """Custom head forward that returns 9 separate outputs"""
        # Ensure x is a list
        if not isinstance(x, (list, tuple)):
            x = [x]
            
        outputs = []
        
        # Check head type and mode
        is_pose26 = hasattr(head, 'cv4_kpts')
        use_end2end = hasattr(head, 'one2one_cv4_kpts')
        
        if use_end2end:
            # end2end=True: use one2one heads
            box_layers = head.one2one_cv2
            cls_layers = head.one2one_cv3
            pose_layers = head.one2one_cv4 if hasattr(head, 'one2one_cv4') else None
            kpts_layers = head.one2one_cv4_kpts if hasattr(head, 'one2one_cv4_kpts') else None
        elif is_pose26:
            # end2end=False, Pose26: use regular cv4/cv4_kpts
            box_layers = head.cv2
            cls_layers = head.cv3
            pose_layers = head.cv4 if hasattr(head, 'cv4') else None
            kpts_layers = head.cv4_kpts if hasattr(head, 'cv4_kpts') else None
        else:
            # Pose (not Pose26): cv4直接输出kpts
            box_layers = head.cv2
            cls_layers = head.cv3
            pose_layers = head.cv4 if hasattr(head, 'cv4') else None
            kpts_layers = None

        for i in range(head.nl):
            # 1. Box 分支
            bboxes = box_layers[i](x[i])
            outputs.append(bboxes)
            
            # 2. Cls 分支
            scores = cls_layers[i](x[i])
            outputs.append(scores)
            
            # 3. Kpt 分支 - 输出原始kpts，NPU端解码
            if kpts_layers is not None:
                # Pose26: cv4 -> cv4_kpts 两步
                pose_feat = pose_layers[i](x[i])
                kpts = kpts_layers[i](pose_feat)
                # 直接输出原始kpts，不解码
                outputs.append(kpts)
            elif pose_layers is not None:
                # Pose: cv4直接输出kpts
                kpts = pose_layers[i](x[i])
                # 直接输出原始kpts，不解码
                outputs.append(kpts)
            else:
                bs = x[i].shape[0]
                h, w = x[i].shape[2], x[i].shape[3]
                outputs.append(torch.zeros(bs, head.nk, h, w, device=x[i].device))
        
        return tuple(outputs)


def export_separate_pose_onnx(model_path, output_name=None, imgsz=[352, 640], opset=11, simplify=True):
    """
    Export YOLO26-face Pose model with separate box, cls, kpt outputs (9 outputs total).
    Supports both end2end=True (Pose26) and end2end=False (Pose) modes.
    Keypoints output raw values, NPU needs to decode.
    """
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)
    
    if hasattr(model.model, 'model'):
        pose_model = model.model
    else:
        pose_model = model
    
    wrapper = PoseSeparateExporter(pose_model)
    wrapper.eval()
    
    # Detect head type and mode
    head = pose_model.model[-1]
    is_pose26 = hasattr(head, 'cv4_kpts')
    use_end2end = hasattr(head, 'one2one_cv4_kpts')
    
    print(f"✓ Created separate output wrapper")
    if is_pose26:
        print(f"✓ Head type: Pose26")
        print(f"✓ Mode: {'end2end=True' if use_end2end else 'end2end=False'}")
    else:
        print(f"✓ Head type: Pose")
    
    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]
    
    print(f"\nStarting export with imgsz={imgsz}, opset={opset}...")
    
    dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1])
    
    weight_path = Path(model_path)
    if output_name is None:
        output_name = str(weight_path.parent / f"{weight_path.stem}_separate.onnx")
    
    output_names = [
        'stride8_box', 'stride8_cls', 'stride8_kpt',
        'stride16_box', 'stride16_cls', 'stride16_kpt',
        'stride32_box', 'stride32_cls', 'stride32_kpt'
    ]
    
    print(f"\nExporting to {output_name}...")
    print(f"Output names: {output_names}")
    
    try:
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
        print(f"\nOutput format (9 outputs):")
        print(f"  Stride 8:")
        print(f"    - stride8_box:  [batch, 4*reg_max, H8, W8]")
        print(f"    - stride8_cls:  [batch, nc, H8, W8]")
        print(f"    - stride8_kpt:  [batch, nk, H8, W8] (raw, needs decode)")
        print(f"  Stride 16:")
        print(f"    - stride16_box: [batch, 4*reg_max, H16, W16]")
        print(f"    - stride16_cls: [batch, nc, H16, W16]")
        print(f"    - stride16_kpt: [batch, nk, H16, W16] (raw, needs decode)")
        print(f"  Stride 32:")
        print(f"    - stride32_box: [batch, 4*reg_max, H32, W32]")
        print(f"    - stride32_cls: [batch, nc, H32, W32]")
        print(f"    - stride32_kpt: [batch, nk, H32, W32] (raw, needs decode)")
        
        if is_pose26:
            if use_end2end:
                print(f"\n⚠️  Pose26 (end2end=True) - NPU decode:")
                print(f"   kpt_x = (raw_x + grid_x + 0.5) * stride")
                print(f"   kpt_y = (raw_y + grid_y + 0.5) * stride")
            else:
                print(f"\n⚠️  Pose26 (end2end=False) - NPU decode:")
                print(f"   kpt_x = (raw_x * 2.0 + grid_x) * stride")
                print(f"   kpt_y = (raw_y * 2.0 + grid_y) * stride")
        else:
            print(f"\n⚠️  Pose - NPU decode:")
            print(f"   kpt_x = (raw_x * 2.0 + grid_x) * stride")
            print(f"   kpt_y = (raw_y * 2.0 + grid_y) * stride")
        
        print(f"\n📝 Note: Keypoints need sigmoid for visibility:")
        print(f"   kpt_vis = sigmoid(raw_vis)")
        
        # Simplify if requested
        if simplify:
            try:
                import onnxslim
                print(f"\nSimplifying ONNX model...")
                onnx_model = onnxslim.slim(output_name)
                onnx_model.save(str(output_name))
                print(f"✓ Model slimmed")
            except Exception as e:
                print(f"Warning: Could not slim model: {e}")
        
        return output_name
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export YOLO26-face Pose model with separate branches')
    parser.add_argument('--weights', type=str, default='runs/pose/train/weights/best.pt',
                       help='Path to model weights (.pt file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ONNX file name (default: auto-generated)')
    parser.add_argument('--imgsz', type=int, nargs=2, default=[288, 512],
                       help='Input image size [height, width]')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify ONNX model')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Disable ONNX simplification')
    
    args = parser.parse_args()
    
    simplify = args.simplify and not args.no_simplify
    
    export_separate_pose_onnx(
        model_path=args.weights,
        output_name=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=simplify
    )
