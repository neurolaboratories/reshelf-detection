import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from random import shuffle
from typing import Dict, List, Tuple
import argparse
import sys
from itertools import chain

import numpy as np
from dateutil.parser import parse as dt_parse



def run(args=None):
    parser = argparse.ArgumentParser(
        description=("Postprocessing script that filters annotations"))

    subparsers = parser.add_subparsers(title="commands",
                                       dest="cmd",
                                       required=True)

    sp = subparsers.add_parser(
        "filter",
        help=
        ("Filter annotations from a coco file based on visible area,"
         "standard deviation distance or absolute pixel areas."
    ))
    sp.add_argument(
        "-c",
        "--coco-file",
        type=str,
        default=None,
        help=("Path to the coco file, i.e. ~/reshelf-detection/coco.json"))
    sp.add_argument(
        "--visible-percentage-threshold",
        type=float,
        default=None,
        help="remove all annotations with visible percentage < threshold")
    sp.add_argument(
        "--std-threshold",
        type=float,
        default=None,
        help=
        ("remove all annotations that have the area smaller than std-threshold"
         "times away from the mean. The statistics are computed per class"))
    sp.add_argument(
        "--area-threshold",
        type=int,
        default=None,
        help=("remove all annotations that have the area < area-threshold"))

    cmd_dict = {"filter": filter_cmd}
    args = parser.parse_args(args=args)
    cmd_dict[args.cmd](args)



def filter_cmd(args):
    coco_file = Path(args.coco_file)
    if not coco_file.is_file():
        print(f"ERROR: File {coco_file} does not exist. Exiting.")
        sys.exit(1)

    coco_out = filter_coco(
        coco_file=coco_file,
        visible_percentage_threshold=args.visible_percentage_threshold,
        std_threshold=args.std_threshold,
        area_threshold=args.area_threshold)

    print(f"Filtered coco saved at {str(coco_out)}")


def filter_coco(coco_file: Path, visible_percentage_threshold: float,
                std_threshold: float, area_threshold: float) -> Path:
    """Filters annotations from a coco based on:
        - visible percentage of the annotation (zia syn specific coco field)
        - that have the area smaller than std-threshold times away from the
          mean. Stats are computed per class
        - annotations that the the absolute area < area threshold
        If any of the args is None, it skips that filtering step. 
        Creates a new file in the same dir as coco file with the filtered
        annotations and returns its path
    """
    coco_out = coco_file.parents[
        0] / f"coco_vp_{visible_percentage_threshold}_std_{std_threshold}_area_{area_threshold}.json"
    with coco_file.open() as f:
        coco = json.load(f)
    filt_ann = coco["annotations"]

    if visible_percentage_threshold:
        filt_ann = _filter_annotations_by_visible_percentage(
            filt_ann, visible_percentage_threshold)

    if std_threshold:
        filt_ann = _filter_annotations_by_std(filt_ann,
                                              visible_percentage_threshold)

    if area_threshold:
        filt_ann = _filter_annotations_by_area(filt_ann, area_threshold)

    coco["annotations"] = filt_ann
    json.dump(coco, coco_out.open("w"), indent=2)

    return coco_out


def _filter_annotations_by_visible_percentage(
        annotations: list, visible_percentage_threshold: float) -> list:
    return [
        ann for ann in annotations
        if ann["visible_percentage"] > visible_percentage_threshold
    ]


def _filter_annotations_by_std(annotations: list,
                               std_threshold: float) -> list:
    areas_per_class = defaultdict(list)
    stats_per_class = {}
    filt_annotations = []

    for ann in annotations:
        areas_per_class[ann["category_id"]].append(
            get_poly_area(ann["segmentation"]))

    for category, areas in areas_per_class.items():
        stats_per_class[category] = {
            "mean": np.mean(areas),
            "std": np.std(areas)
        }

    for ann in annotations:
        category = ann["category_id"]
        mean, std = stats_per_class[category]["mean"], stats_per_class[
            category]["std"]
        if ann["area"] > mean - (std * std_threshold):
            filt_annotations.append(ann)

    return filt_annotations


def _filter_annotations_by_area(annotations: list,
                                area_threshold: int) -> list:

    return [
        ann for ann in annotations
        if get_poly_area(ann["segmentation"]) > area_threshold
    ]


def get_poly_area(mask: list) -> float:
    """ Compute area of a closed polygon using the shoelace algorithm.
        Sum up all the closed polygons areas to get the total area
        https://en.wikipedia.org/wiki/Shoelace_formula
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    """
    for polyline in mask:
        x = [polyline[i] for i in range(0, len(polyline), 2)]
        y = [polyline[i + 1] for i in range(0, len(polyline), 2)]
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

if __name__ == "__main__":
    run()

