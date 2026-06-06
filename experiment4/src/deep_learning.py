import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# COCO 2017 类别标签 (部分常见类别展示)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_mask_rcnn_model(device):
    """加载预训练的 Mask R-CNN 模型"""
    print(">>> 正在加载 Mask R-CNN 预训练模型 (初次运行需下载约160MB权重)...")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights, progress=True)
    model.to(device)
    model.eval()
    return model


def predict_and_visualize(model, img_bgr, device, confidence_threshold=0.5):
    """模型推理并可视化结果 (绘制 Mask, Box, Label)"""
    # 图像预处理: BGR -> RGB -> Tensor [0,1] -> Batch [1, C, H, W]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torchvision.transforms.functional.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    # 解析预测结果
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    masks = prediction['masks'].cpu().numpy()

    # 绘制可视化图像
    vis_img = img_rgb.copy()
    num_valid_detections = 0

    # 随机生成掩码颜色
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

    for i in range(len(scores)):
        if scores[i] > confidence_threshold:
            num_valid_detections += 1
            box = boxes[i].astype(int)
            class_id = labels[i]
            score = scores[i]
            mask = masks[i, 0] > 0.5  # 二值化 Mask

            color = colors[class_id].tolist()

            # 1. 绘制半透明 Mask
            colored_mask = np.zeros_like(vis_img, dtype=np.uint8)
            colored_mask[mask] = color
            vis_img = cv2.addWeighted(vis_img, 1.0, colored_mask, 0.5, 0)

            # 2. 绘制 Bounding Box
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, 2)

            # 3. 绘制标签和置信度
            label_name = COCO_INSTANCE_CATEGORY_NAMES[class_id] if class_id < len(
                COCO_INSTANCE_CATEGORY_NAMES) else 'Unknown'
            text = f"{label_name}: {score:.2f}"
            cv2.putText(vis_img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return vis_img, num_valid_detections


def add_gaussian_noise(img_bgr, mean=0, sigma=15):
    """为现场测评添加高斯噪声"""
    noise = np.random.normal(mean, sigma, img_bgr.shape)
    noisy_img = img_bgr.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)