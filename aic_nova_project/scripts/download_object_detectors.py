import os
import subprocess
import shutil

def download_yolo_world():
    print("Downloading YOLO-World (yolov8s-world.pt)...")
    try:
        from ultralytics import YOLOWorld
        model = YOLOWorld("yolov8s-world.pt")
        if os.path.exists("yolov8s-world.pt"):
            shutil.move("yolov8s-world.pt", "weights/yolov8s-world.pt")
        print("YOLO-World downloaded.")
    except Exception as e:
        print(f"Error downloading YOLO-World: {e}")

def download_codetr():
    print("Downloading Co-DETR variants using mim...")
    try:
        # ResNet-50
        subprocess.run([
            "mim", "download", "mmdet", 
            "--config", "co_dino_5scale_r50_8xb2_1x_coco",
            "--dest", "weights/"
        ], check=True)
        # Swin-L
        subprocess.run([
            "mim", "download", "mmdet", 
            "--config", "co_dino_5scale_swin_l_16e_o365tococo",
            "--dest", "weights/"
        ], check=True)
        print("Co-DETR weights downloaded.")
    except Exception as e:
        print(f"Error downloading Co-DETR: {e}")

if __name__ == "__main__":
    os.makedirs("weights", exist_ok=True)
    download_yolo_world()
    download_codetr()
    print("All object detection weights download finished.")
