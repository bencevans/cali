from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
import json
from cali import Cali, plot_result
from rich import print
from rich.console import Console

console = Console()

def parse_args() -> Namespace:
    # parse args, there's detect and visualise subcommands
    parser = ArgumentParser(description="Calibration Pole Detector and Annotator")
    

    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Detect calibration poles in an image")
    detect_parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for detections")
    detect_parser.add_argument("image_source", type=Path, help="Path to the input image")
    detect_parser.add_argument("--recursive", action="store_true", default=False, help="Recurse into subdirectories")
    detect_parser.add_argument("output", type=Path, help="Path to save the full JSON results")
    detect_parser.add_argument("--relative", action="store_true", default=False, help="Store image paths relative to image_source in the JSON output")
    
    plot_parser = subparsers.add_parser("plot", help="Plot detected calibration poles in an image")
    plot_parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for detections")
    plot_parser.add_argument("image_source", type=Path, help="Path to the input image")
    plot_parser.add_argument("--recursive", action="store_true", default=False, help="Recurse into subdirectories")
    
    return parser.parse_args()

def enumerate_images(image_source: Path, recursive: bool = False) -> list[Path]:
    if image_source.is_file():
        return [image_source]
    elif image_source.is_dir():
        if recursive:
            return list(image_source.rglob("*.[jJ][pP][gG]")) + list(image_source.rglob("*.[pP][nN][gG]"))
        else:
            return list(image_source.glob("*.[jJ][pP][gG]")) + list(image_source.glob("*.[pP][nN][gG]"))
    else:
        raise ValueError(f"Invalid image source: {image_source}")

def main() -> None:
    args = parse_args()

    model = Cali()

    if args.command == "detect":
        image_paths = enumerate_images(args.image_source, recursive=args.recursive)
        all_results = []
        base_path = args.image_source if args.image_source.is_dir() else args.image_source.parent
        for result in model.detect_generator_list([str(p) for p in image_paths]):
            n_detections = len(result.detections)
            n_keypoints = sum(len(d.keypoints) for d in result.detections)
            console.print(f"[bold]{Path(result.image_path).name}[/bold]: {n_detections} detection(s), {n_keypoints} keypoint(s)")
            result_dict = asdict(result)
            if args.relative:
                result_dict["image_path"] = str(Path(result.image_path).relative_to(base_path))
            all_results.append(result_dict)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        console.print(f"Results saved to [green]{args.output}[/green]")

    elif args.command == "plot":
        # plot only supports a single image
        if args.image_source.is_dir():
            print("Plotting only supports a single image. Please provide a path to an image file.")
            return

        result = model.detect(str(args.image_source))
        plot_result(result)
   