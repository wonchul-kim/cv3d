from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="/HDD/etc/outputs/calibration_single_tag/object/seg_object.v1i.coco-segmentation/train/",
    save_dir="/HDD/etc/outputs/calibration_single_tag/object/coco2yolo_seg",
    use_keypoints=False,
    use_segments=True,
    cls91to80=False
)