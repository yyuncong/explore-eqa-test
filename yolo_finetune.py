from ultralytics import YOLOWorld



# start from scratch
# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8x-worldv2.pt")
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data="/gpfs/u/home/LMCG/LMCGhazh/scratch/yanghan/explore-eqa-test/yolo_finetune/dataset.yaml",
    epochs=100,
    imgsz=1280,
    batch=8 * 6,
    device=[0, 1, 2, 3, 4, 5]
)



# # resume
# model = YOLOWorld("/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/runs/detect/train2/weights/last.pt")
# results = model.train(
#     data="/gpfs/u/home/LMCG/LMCGhazh/scratch/yanghan/explore-eqa-test/yolo_finetune/dataset.yaml",
#     epochs=100,
#     imgsz=1280,
#     batch=8 * 6,
#     device=[0, 1, 2, 3, 4, 5],
#     resume=True,
# )
