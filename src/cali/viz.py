import matplotlib.pyplot as plt

from cali.data_models import ExtentKeypoint, HeightKeypoint, ImageResult


def _keypoint_label(keypoint: ExtentKeypoint | HeightKeypoint) -> str:
    if isinstance(keypoint, ExtentKeypoint):
        return f"{keypoint.name}: {keypoint.confidence:.2f}"
    return f"{int(keypoint.height * 100)}cm: {keypoint.confidence:.2f}"


def plot_result(result: ImageResult) -> None:
    image = plt.imread(result.image_path)
    plt.imshow(image)

    label_offsets = [
        (10, -14),
        (10, 14),
        (-10, -14),
        (-10, 14),
        (14, 0),
        (-14, 0),
        (0, -16),
        (0, 16),
    ]

    for detection in result.detections:
        x1, y1, x2, y2 = detection.bounding_box
        plt.plot([x1, x2], [y1, y1], "b-")
        plt.plot([x1, x2], [y2, y2], "b-")
        plt.plot([x1, x1], [y1, y2], "b-")
        plt.plot([x2, x2], [y1, y2], "b-")

        plt.text(
            x1 + 20,
            y1 - 20,
            f"{detection.name}: {detection.confidence:.2f}",
            color="white",
            fontsize=12,
            backgroundcolor="blue",
        )

        for kp_idx, keypoint in enumerate(detection.keypoints):
            x, y = keypoint.x, keypoint.y
            plt.plot(x, y, "ro")

            dx, dy = label_offsets[kp_idx % len(label_offsets)]
            plt.annotate(
                _keypoint_label(keypoint),
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                color="white",
                fontsize=10,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox={
                    "facecolor": "red",
                    "alpha": 0.75,
                    "pad": 1.5,
                    "edgecolor": "none",
                },
                arrowprops={"arrowstyle": "-", "color": "red", "lw": 0.8},
            )

    plt.show()
