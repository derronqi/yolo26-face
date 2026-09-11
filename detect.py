"""
YOLO26-Face Detection Script - Simple version using Ultralytics API
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def detect(opt):
    """Main detection function"""
    # Load model
    print(f"Loading model: {opt.weights}")
    model = YOLO(opt.weights)
    
    # Run inference
    print(f"\nProcessing: {opt.source}")
    results = model.predict(
        source=opt.source,
        imgsz=opt.imgsz,
        conf=opt.conf,
        iou=opt.iou,
        device=opt.device,
        save=opt.save,
        save_dir=opt.save_dir if opt.save_dir else None,
        show=opt.show,
        verbose=True
    )
    
    # Print summary
    print(f"\nDetection completed!")
    if opt.save:
        print(f"Results saved to: runs/detect")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO26-Face Detection')
    
    parser.add_argument('--weights', type=str, 
                       default='runs/pose/yolov5s-face/weights/best.pt',
                       help='Model weights path')
    parser.add_argument('--source', type=str, 
                       default='data/widerface/val/images',
                       help='Image file or directory')
    parser.add_argument('--imgsz', type=int, default=640, nargs=2,
                       help='Image size [height, width]')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold')
    parser.add_argument('--device', type=str, default='0',
                       help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--save', action='store_true',
                       help='Save detection results')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Custom save directory')
    parser.add_argument('--show', action='store_true',
                       help='Show results in window')
    
    opt = parser.parse_args()
    detect(opt)
