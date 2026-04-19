from ultralytics import YOLO
model = YOLO("weights/teacher_best.pt")
print(model.names)
# Output ví dụ: {0: 'obstacle', 1: 'furniture', 2: 'step', ...}
# Hoặc nếu dataset dùng COCO subset: {0: 'chair', 1: 'table', ...}