from ultralytics import YOLO


def to_train():
    model = YOLO("yolo11n-seg.pt")

    train_results = model.train(
        data="/HDD/_projects/bitbucket/aivot-rl/aivot_rl/tools/roboflow/cfg/datasets/object_seg_coco.yaml",
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        project='/HDD/etc/outputs/calibration_single_tag/object/outputs'
    )


def to_predict():
    import cv2 
    import numpy as np
    model = YOLO("/HDD/etc/outputs/calibration_single_tag/object/outputs/train4/weights/last.pt")
    
    view = cv2.imread('/HDD/etc/outputs/calibration_single_tag/object/dataset/images/val/cnt_0002.png')
    results = model(view)
    if results and results[0].masks is not None:
        result = results[0]
        boxes = result.boxes
        masks = result.masks
        masks = result.masks.data.cpu().numpy()  # mask in matrix format (N x H x W)
        overlay = view.copy()
        alpha = 0.5  # Transparency factor
        
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)] # Blue, Green, Red, Cyan

        for i, mask in enumerate(masks):
            class_id = 0
            # Ensure the mask is scaled to 255 (it's often 0s and 1s)
            mask_resized = cv2.resize(mask.astype(np.uint8), (view.shape[1], view.shape[0]))
            
            # Get the color for this class, cycling through the defined colors
            mask_color = colors[class_id % len(colors)] 
            
            # Find the indices where the mask is active
            mask_indices = mask_resized > 0
            
            # Apply the color to the overlay
            overlay[mask_indices] = mask_color
            
            box_xyxy = list(map(int, boxes.xyxy[i]))
            cv2.rectangle(view, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (0, 0, 255), 0)
            # cv2.putText(view, t, (x, y_cursor), font, scale, color, thickness, cv2.LINE_AA)
        
        # Blend the overlay with the original image
        view = cv2.addWeighted(overlay, alpha, view, 1 - alpha, 0)

        cv2.imwrite('/HDD/etc/outputs/calibration_single_tag/cal/cal.png', view)

    
if __name__ == '__main__':
    to_predict()