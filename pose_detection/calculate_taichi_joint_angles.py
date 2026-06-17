#!/usr/bin/env python3
"""
Calculate Tai Chi motion features from frame-by-frame 3D joint coordinates.

Input format:
    A CSV or text file with a Frame column and joint columns such as p11, p12, ...
    Each joint cell should look like "(x, y, z)".

Example:
    python calculate_taichi_joint_angles.py
    python calculate_taichi_joint_angles.py frame_coordinates.csv
    python calculate_taichi_joint_angles.py frame_coordinates.csv --fps 30

If no input file is given, the script looks in its own folder for:
    1. frame_data.csv
    2. frame_data.txt

Default outputs:
    frame_data_motion_features.csv
    frame_data_motion_features.txt
    frame_data_angle_codebook.txt
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path
from typing import Iterable


# Angles are defined as (point_a, vertex_point, point_c).
# The middle joint is the vertex of the angle.
ANGLES_OF_INTEREST = [
    (11, 13, 15),
    (12, 14, 16),
    (11, 13, 23),
    (12, 14, 24),
    (11, 23, 25),
    (12, 24, 26),
]
ANGLE_NUMBERS = {
    angle: index
    for index, angle in enumerate(ANGLES_OF_INTEREST, start=1)
}

# Left/right angle pairs for symmetry. Each output is abs(left - right).
SYMMETRY_PAIRS = [
    ((11, 13, 15), (12, 14, 16)),
    ((11, 13, 23), (12, 14, 24)),
    ((11, 23, 25), (12, 24, 26)),
]


def parse_point(value: str) -> tuple[float, float, float] | None:
    """Parse a coordinate cell like '(0.1, 0.2, -0.3)'."""
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    try:
        point = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None

    if not isinstance(point, (tuple, list)) or len(point) != 3:
        return None

    try:
        return float(point[0]), float(point[1]), float(point[2])
    except (TypeError, ValueError):
        return None


def parse_frame(value: str, fallback: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def vector_from_vertex(
    point: tuple[float, float, float],
    vertex: tuple[float, float, float],
) -> tuple[float, float, float]:
    return point[0] - vertex[0], point[1] - vertex[1], point[2] - vertex[2]


def vector_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def point_distance(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
) -> float:
    return vector_norm(
        (
            point_a[0] - point_b[0],
            point_a[1] - point_b[1],
            point_a[2] - point_b[2],
        )
    )


def angle_degrees(
    point_a: tuple[float, float, float] | None,
    vertex: tuple[float, float, float] | None,
    point_c: tuple[float, float, float] | None,
) -> float | None:
    """Return angle ABC in degrees, where B is the vertex."""
    if point_a is None or vertex is None or point_c is None:
        return None

    ba = vector_from_vertex(point_a, vertex)
    bc = vector_from_vertex(point_c, vertex)
    norm_ba = vector_norm(ba)
    norm_bc = vector_norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    cosine = dot_product / (norm_ba * norm_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def format_number(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def angle_column_name(angle: tuple[int, int, int]) -> str:
    return f"angle_{ANGLE_NUMBERS[angle]}_deg"


def angular_velocity_column_name(angle: tuple[int, int, int]) -> str:
    return f"angular_velocity_{ANGLE_NUMBERS[angle]}_deg_per_s"


def rom_column_name(angle: tuple[int, int, int], window_seconds: float) -> str:
    label = f"{window_seconds:g}s".replace(".", "p")
    return f"rom_{label}_angle_{ANGLE_NUMBERS[angle]}_deg"


def vertex_speed_column_name(angle: tuple[int, int, int]) -> str:
    return (
        f"vertex_speed_angle_{ANGLE_NUMBERS[angle]}"
        f"_p{angle[1]}_coord_per_s"
    )


def symmetry_column_name(
    left_angle: tuple[int, int, int],
    right_angle: tuple[int, int, int],
) -> str:
    return (
        f"symmetry_absdiff_angle_{ANGLE_NUMBERS[left_angle]}"
        f"_vs_{ANGLE_NUMBERS[right_angle]}_deg"
    )


def read_records(input_csv: Path, fps: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    needed_joints = sorted(
        {joint for angle in ANGLES_OF_INTEREST for joint in angle}
    )

    with input_csv.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Input file does not have a CSV header row.")

        for row_index, row in enumerate(reader):
            frame_text = row.get("Frame", "").strip()
            frame_number = parse_frame(frame_text, row_index)
            points = {
                joint: parse_point(row.get(f"p{joint}", ""))
                for joint in needed_joints
            }
            angles = {
                angle: angle_degrees(
                    points[angle[0]],
                    points[angle[1]],
                    points[angle[2]],
                )
                for angle in ANGLES_OF_INTEREST
            }

            records.append(
                {
                    "frame_text": frame_text or str(row_index),
                    "frame_number": frame_number,
                    "time_sec": frame_number / fps,
                    "points": points,
                    "angles": angles,
                }
            )

    return records


def previous_delta_time(
    records: list[dict[str, object]],
    index: int,
) -> float | None:
    if index == 0:
        return None

    current_time = records[index]["time_sec"]
    previous_time = records[index - 1]["time_sec"]
    if not isinstance(current_time, float) or not isinstance(previous_time, float):
        return None

    delta_time = current_time - previous_time
    return delta_time if delta_time > 0 else None


def calculate_velocity(
    current_value: float | None,
    previous_value: float | None,
    delta_time: float | None,
) -> float | None:
    if current_value is None or previous_value is None or delta_time is None:
        return None
    return (current_value - previous_value) / delta_time


def calculate_speed(
    current_point: tuple[float, float, float] | None,
    previous_point: tuple[float, float, float] | None,
    delta_time: float | None,
) -> float | None:
    if current_point is None or previous_point is None or delta_time is None:
        return None
    return point_distance(current_point, previous_point) / delta_time


def calculate_rolling_rom(
    records: list[dict[str, object]],
    index: int,
    angle: tuple[int, int, int],
    window_seconds: float,
) -> float | None:
    current_time = records[index]["time_sec"]
    if not isinstance(current_time, float):
        return None

    start_time = current_time - window_seconds
    values: list[float] = []

    for record in records:
        record_time = record["time_sec"]
        if not isinstance(record_time, float):
            continue
        if start_time <= record_time <= current_time:
            angles = record["angles"]
            if isinstance(angles, dict):
                value = angles.get(angle)
                if isinstance(value, float):
                    values.append(value)

    if not values:
        return None
    return max(values) - min(values)

def calculate_rolling_rom_optimized(
    records: list[dict[str, object]],
    angle: tuple[int, int, int],
    window_seconds: float,
) -> list[float | None]:
    results: list[float | None] = [None] * len(records)
    window_values: list[float] = []
    window_start_index = 0
    
    for i, record in enumerate(records):
        current_time = record["time_sec"]
        if not isinstance(current_time, float):
            continue
            
        # Add current value
        angles = record.get("angles")
        if isinstance(angles, dict):
            value = angles.get(angle)
            if isinstance(value, float):
                window_values.append(value)
        
        # Remove values outside the window
        while window_start_index <= i:
            if not isinstance(records[window_start_index]["time_sec"], float):
                window_start_index += 1
                continue
                
            if records[window_start_index]["time_sec"] < current_time - window_seconds:
                # Remove this record's value if it was included
                old_angles = records[window_start_index].get("angles")
                if isinstance(old_angles, dict):
                    old_value = old_angles.get(angle)
                    if isinstance(old_value, float):
                        # Remove the first occurrence (assuming values are in order)
                        try:
                            window_values.remove(old_value)
                        except ValueError:
                            pass  # Already removed or not found
                window_start_index += 1
            else:
                break
        
        if window_values:
            results[i] = max(window_values) - min(window_values)
    
    return results

def build_feature_rows(
    records: list[dict[str, object]],
    window_seconds: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    
    # Pre-calculate ROMs for all angles at once (O(n) each)
    rom_values = {}
    for angle in ANGLES_OF_INTEREST:
        rom_values[angle] = calculate_rolling_rom_optimized(records, angle, window_seconds)

    for index, record in enumerate(records):
        output_row: dict[str, str] = {
            "Frame": str(record["frame_text"]),
            "time_sec": format_number(record["time_sec"]),
        }

        angles = record["angles"]
        points = record["points"]
        previous_record = records[index - 1] if index > 0 else None
        delta_time = previous_delta_time(records, index)

        if not isinstance(angles, dict) or not isinstance(points, dict):
            continue

        # Add angles
        for angle in ANGLES_OF_INTEREST:
            output_row[angle_column_name(angle)] = format_number(angles.get(angle))

        # Add angular velocities
        for angle in ANGLES_OF_INTEREST:
            previous_angle = None
            if previous_record and isinstance(previous_record["angles"], dict):
                previous_angle = previous_record["angles"].get(angle)
            angular_velocity = calculate_velocity(
                angles.get(angle),
                previous_angle,
                delta_time,
            )
            output_row[angular_velocity_column_name(angle)] = format_number(
                angular_velocity
            )

        # Add vertex speeds
        for angle in ANGLES_OF_INTEREST:
            vertex_joint = angle[1]
            previous_point = None
            if previous_record and isinstance(previous_record["points"], dict):
                previous_point = previous_record["points"].get(vertex_joint)
            output_row[vertex_speed_column_name(angle)] = format_number(
                calculate_speed(points.get(vertex_joint), previous_point, delta_time)
            )

        # Add ROMs from cache
        for angle in ANGLES_OF_INTEREST:
            output_row[rom_column_name(angle, window_seconds)] = format_number(
                rom_values[angle][index]
            )

        # Add symmetry values
        for left_angle, right_angle in SYMMETRY_PAIRS:
            left_value = angles.get(left_angle)
            right_value = angles.get(right_angle)
            symmetry = (
                abs(left_value - right_value)
                if isinstance(left_value, float) and isinstance(right_value, float)
                else None
            )
            output_row[symmetry_column_name(left_angle, right_angle)] = format_number(
                symmetry
            )

        rows.append(output_row)

    return rows

def output_columns(window_seconds: float) -> list[str]:
    columns = ["Frame", "time_sec"]
    columns += [angle_column_name(angle) for angle in ANGLES_OF_INTEREST]
    columns += [angular_velocity_column_name(angle) for angle in ANGLES_OF_INTEREST]
    columns += [vertex_speed_column_name(angle) for angle in ANGLES_OF_INTEREST]
    columns += [rom_column_name(angle, window_seconds) for angle in ANGLES_OF_INTEREST]
    columns += [
        symmetry_column_name(left_angle, right_angle)
        for left_angle, right_angle in SYMMETRY_PAIRS
    ]
    return columns


def calculate_motion_features(
    input_csv: Path,
    fps: float,
    window_seconds: float,
) -> list[dict[str, str]]:
    records = read_records(input_csv, fps)
    return build_feature_rows(records, window_seconds)


def write_csv(
    rows: Iterable[dict[str, str]],
    output_csv: Path,
    columns: list[str],
) -> None:
    rows = list(rows)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_text(
    rows: Iterable[dict[str, str]],
    output_txt: Path,
    columns: list[str],
) -> None:
    rows = list(rows)
    widths = {
        column: max(
            len(column),
            *(len(row.get(column, "")) for row in rows),
        )
        for column in columns
    }

    with output_txt.open("w", encoding="utf-8") as file:
        file.write("Tai Chi motion features\n")
        file.write("Angles are in degrees. Angular velocity is deg/s.\n")
        file.write("Vertex speed is coordinate units/s for each angle's middle joint.\n")
        file.write("ROM is rolling range of motion over the selected time window.\n\n")
        file.write(" | ".join(column.ljust(widths[column]) for column in columns))
        file.write("\n")
        file.write("-+-".join("-" * widths[column] for column in columns))
        file.write("\n")

        for row in rows:
            file.write(
                " | ".join(row.get(column, "").ljust(widths[column]) for column in columns)
            )
            file.write("\n")


def write_angle_codebook(output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        file.write("Angle codebook\n")
        file.write("Format: angle number = (point_a, vertex_point, point_c)\n")
        file.write("The middle point is the angle vertex.\n\n")

        for angle in ANGLES_OF_INTEREST:
            file.write(f"angle_{ANGLE_NUMBERS[angle]} = {angle}\n")

        file.write("\nVertex speed variables\n")
        for angle in ANGLES_OF_INTEREST:
            file.write(
                f"{vertex_speed_column_name(angle)} = speed of p{angle[1]}, "
                f"the vertex joint of angle_{ANGLE_NUMBERS[angle]}\n"
            )

        file.write("\nSymmetry variables\n")
        for left_angle, right_angle in SYMMETRY_PAIRS:
            file.write(
                f"{symmetry_column_name(left_angle, right_angle)} = "
                f"abs(angle_{ANGLE_NUMBERS[left_angle]} - "
                f"angle_{ANGLE_NUMBERS[right_angle]})\n"
            )


def default_output_path(input_csv: Path, suffix: str) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_motion_features{suffix}")


def default_codebook_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_angle_codebook.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate selected 3D joint angles and motion features.",
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        help="CSV/text file containing Frame and p11/p12/... coordinate columns.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frame rate. Default: 30.",
    )
    parser.add_argument(
        "--rom-window-seconds",
        type=float,
        default=1.0,
        help="Rolling ROM time window in seconds. Default: 1.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output CSV path. Default: INPUT_motion_features.csv",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        help="Output readable text path. Default: INPUT_motion_features.txt",
    )
    return parser.parse_args()


def find_default_input_file() -> Path:
    script_folder = Path(__file__).resolve().parent
    candidates = [
        script_folder / "frame_data.csv",
        script_folder / "frame_data.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_names = ", ".join(candidate.name for candidate in candidates)
    raise FileNotFoundError(
        "No input file was provided, and no default input file was found.\n"
        f"Put {candidate_names} in the same folder as this script, or provide an input path."
    )


def main() -> None:
    args = parse_args()
    input_csv = (
        args.input_csv.expanduser().resolve()
        if args.input_csv
        else find_default_input_file()
    )

    if args.fps <= 0:
        raise ValueError("FPS must be greater than 0.")
    if args.rom_window_seconds <= 0:
        raise ValueError("ROM window seconds must be greater than 0.")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    output_csv = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(input_csv, ".csv")
    )
    output_txt = (
        args.txt.expanduser().resolve()
        if args.txt
        else default_output_path(input_csv, ".txt")
    )
    output_codebook = default_codebook_path(input_csv)

    columns = output_columns(args.rom_window_seconds)
    rows = calculate_motion_features(
        input_csv,
        args.fps,
        args.rom_window_seconds,
    )
    write_csv(rows, output_csv, columns)
    write_text(rows, output_txt, columns)
    write_angle_codebook(output_codebook)

    print(f"Input file: {input_csv}")
    print(f"FPS: {args.fps:g}")
    print(f"ROM window: {args.rom_window_seconds:g} seconds")
    print(f"Processed {len(rows)} frames.")
    print(f"CSV output: {output_csv}")
    print(f"Text output: {output_txt}")
    print(f"Angle codebook: {output_codebook}")


if __name__ == "__main__":
    main()
