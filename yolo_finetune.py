from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8x-worldv2.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/gpfs/u/home/LMCG/LMCGhazh/scratch/yanghan/explore-eqa-test/yolo_finetune/dataset.yaml", epochs=100, imgsz=1280, batch=8 * 6, device=[0, 1, 2, 3, 4, 5])

