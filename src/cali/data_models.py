from dataclasses import dataclass
from typing import Literal, List


@dataclass
class ExtentKeypoint:
    """Keypoint representing the extent of the pole (e.g., base or top)."""

    name: Literal["base", "top"]
    """
    The name of the keypoint, which indicates whether it represents the base or the top of the pole.
    """

    x: float
    """The x-coordinate of the keypoint in the image."""

    y: float
    """The y-coordinate of the keypoint in the image."""

    confidence: float
    """The confidence score of the keypoint prediction, between 0 and 1."""


@dataclass
class HeightKeypoint:
    """Keypoint representing a height measurement along the pole (e.g., 20cm, 40cm, etc.)."""

    name: Literal["height"]
    """The name of the keypoint, which indicates that it represents a height measurement along the pole (e.g., 20cm, 40cm, etc.)."""

    x: float
    """The x-coordinate of the keypoint in the image."""

    y: float
    """The y-coordinate of the keypoint in the image."""

    height: float
    """The height value of the keypoint in meters from the base of the pole."""

    confidence: float
    """The confidence score of the keypoint prediction, between 0 and 1."""


Keypoint = ExtentKeypoint | HeightKeypoint
"""A union type that can represent either an extent keypoint (base or top) or a height keypoint (20cm, 40cm, etc.)."""



@dataclass
class Detection:
    """A data class representing a detected calibration pole in an image."""

    confidence: float
    """The confidence score of the detection, between 0 and 1."""

    name: Literal["calibration_pole"]
    """The name of the detected object, which is always 'calibration_pole'."""

    bounding_box: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    """The bounding box of the detected pole, represented as a tuple of (x1, y1, x2, y2) coordinates."""

    keypoints: List[Keypoint]
    """A list of keypoints associated with the detected pole, which can include both extent keypoints (base and top) and height keypoints (20cm, 40cm, etc.)."""


@dataclass
class ImageResult:
    """A data class representing the detection results for a single image."""

    image_path: str
    """The file path of the input image."""

    width: int
    """The width of the input image in pixels."""

    height: int
    """The height of the input image in pixels."""

    detections: List[Detection]
    """A list of detections found in the image."""