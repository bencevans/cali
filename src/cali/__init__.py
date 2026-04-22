from cali.data_models import (
    Detection,
    ExtentKeypoint,
    HeightKeypoint,
    ImageResult,
    Keypoint,
)
from cali.viz import plot_result as plot_result
from typing import Iterator, List
from ultralytics import YOLO


_KEYPOINT_INDEX_TO_NAME = [
    "base",
    "20cm",
    "40cm",
    "60cm",
    "80cm",
    "100cm",
    "top",
]


class Cali:
    """
    Calibration Pole Detector and Annotator

    Example usage:
        from cali import Cali, plot_result

        model = Cali()

        image_path = "/path/to/image.jpg"
        result = model.detect(image_path)

        plot_result(result)
    """

    def __init__(self, conf_threshold: float = 0.5) -> None:
        self.model = YOLO("/Users/ben/Downloads/v0-yolo26n-pose-best.pt")
        self.conf_threshold = conf_threshold

    def detect(self, image_path: str) -> ImageResult:
        results = self.detect_list([image_path])
        return results[0]

    def detect_list(self, image_paths: List[str]) -> List[ImageResult]:
        return list(self.detect_generator_list(image_paths))

    def detect_generator_list(self, image_paths: List[str]) -> Iterator[ImageResult]:
        for image_path in image_paths:
            (r,) = self.model([image_path], conf=self.conf_threshold)

            image_detections: List[Detection] = []
            keypoints_xy = (
                r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
            )
            keypoints_conf = (
                r.keypoints.conf.cpu().numpy()
                if r.keypoints is not None and r.keypoints.conf is not None
                else None
            )

            for box_idx, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                conf = box.conf.cpu().numpy()[0]
                keypoints: List[Keypoint] = []
                if keypoints_xy is not None and box_idx < len(keypoints_xy):
                    points = keypoints_xy[box_idx]
                    point_confs = (
                        keypoints_conf[box_idx]
                        if keypoints_conf is not None and box_idx < len(keypoints_conf)
                        else [None] * len(points)
                    )

                    for kp_idx, ((x, y), kp_conf) in enumerate(
                        zip(points, point_confs)
                    ):
                        if x == 0 and y == 0:
                            continue
                        if kp_conf is not None and kp_conf <= self.conf_threshold:
                            continue

                        kp_name = (
                            _KEYPOINT_INDEX_TO_NAME[kp_idx]
                            if kp_idx < len(_KEYPOINT_INDEX_TO_NAME)
                            else f"kp_{kp_idx}"
                        )

                        if kp_name == "base":
                            keypoints.append(
                                ExtentKeypoint(
                                    name="base",
                                    x=float(x),
                                    y=float(y),
                                    confidence=float(kp_conf),
                                )
                            )
                        elif kp_name == "top":
                            keypoints.append(
                                ExtentKeypoint(
                                    name="top",
                                    x=float(x),
                                    y=float(y),
                                    confidence=float(kp_conf),
                                )
                            )
                        elif kp_name.endswith("cm") and kp_name[:-2].isdigit():
                            height_m = float(kp_name[:-2]) / 100.0
                            keypoints.append(
                                HeightKeypoint(
                                    name="height",
                                    x=float(x),
                                    y=float(y),
                                    height=height_m,
                                    confidence=float(kp_conf),
                                )
                            )

                image_detections.append(
                    Detection(
                        confidence=float(conf),
                        name="calibration_pole",
                        bounding_box=(float(x1), float(y1), float(x2), float(y2)),
                        keypoints=keypoints,
                    )
                )

            orig_h, orig_w = r.orig_img.shape[:2]
            yield ImageResult(
                image_path=image_path,
                width=int(orig_w),
                height=int(orig_h),
                detections=image_detections,
            )


def main() -> None:
    from cali.cli import main

    main()
