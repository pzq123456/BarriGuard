from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    SETTINGS["tensorboard"] = True
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo11m_seg_barrier_{current_time}"
    
    # 使用分割模型（可根据需要选择n/s/m/l/x版本）
    model = YOLO("yolo26m-seg.pt")

    model.train(
        data="dataset/Barrier.v2i.yolov11/data.yaml",  # 您的数据路径
        epochs=300,
        patience=50,
        imgsz=640,  # 分割任务建议使用较大尺寸
        batch=8,   # 分割任务内存占用更大，适当减小batch
        device=0,   # 使用GPU，如果有多个GPU可设为 [0,1,2,3]
        name=run_name,
        workers=4,
        amp=True,   # 分割任务可开启混合精度训练
    )

if __name__ == "__main__":
    main()