import ast
import math
import pandas as pd


DEFAULT_ANGLES_OF_INTEREST = [
    (11, 13, 15),
    (12, 14, 16),
    (11, 13, 23),
    (12, 14, 24),
    (11, 23, 25),
    (12, 24, 26),
]

DEFAULT_SYMMETRY_PAIRS = [
    ((11, 13, 15), (12, 14, 16)),
    ((11, 13, 23), (12, 14, 24)),
    ((11, 23, 25), (12, 24, 26)),
]


def parse_point(value):
    if value is None or pd.isna(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = ast.literal_eval(value)

    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None

    return tuple(float(v) for v in value)


def vector_from_vertex(point, vertex):
    return (
        point[0] - vertex[0],
        point[1] - vertex[1],
        point[2] - vertex[2],
    )


def vector_norm(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def point_distance(point_a, point_b):
    return vector_norm(
        (
            point_a[0] - point_b[0],
            point_a[1] - point_b[1],
            point_a[2] - point_b[2],
        )
    )


def angle_degrees(point_a, vertex, point_c):
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


def make_angle_numbers(angles_of_interest):
    return {
        angle: index
        for index, angle in enumerate(angles_of_interest, start=1)
    }


def angle_column_name(angle, angle_numbers):
    return f"angle_{angle_numbers[angle]}_deg"


def angular_velocity_column_name(angle, angle_numbers):
    return f"angular_velocity_{angle_numbers[angle]}_deg_per_s"


def rom_column_name(angle, angle_numbers, window_seconds):
    label = f"{window_seconds:g}s".replace(".", "p")
    return f"rom_{label}_angle_{angle_numbers[angle]}_deg"


def vertex_speed_column_name(angle, angle_numbers):
    return (
        f"vertex_speed_angle_{angle_numbers[angle]}"
        f"_p{angle[1]}_coord_per_s"
    )


def symmetry_column_name(left_angle, right_angle, angle_numbers):
    return (
        f"symmetry_absdiff_angle_{angle_numbers[left_angle]}"
        f"_vs_{angle_numbers[right_angle]}_deg"
    )


def calculate_velocity(current_value, previous_value, delta_time):
    if current_value is None or previous_value is None or delta_time is None:
        return None
    return (current_value - previous_value) / delta_time


def calculate_speed(current_point, previous_point, delta_time):
    if current_point is None or previous_point is None or delta_time is None:
        return None
    return point_distance(current_point, previous_point) / delta_time


def calculate_rolling_rom(values, times, window_seconds):
    results = []

    for i, current_time in enumerate(times):
        start_time = current_time - window_seconds

        window_values = [
            value
            for value, time in zip(values, times)
            if value is not None and start_time <= time <= current_time
        ]

        if len(window_values) == 0:
            results.append(None)
        else:
            results.append(max(window_values) - min(window_values))

    return results


def calculate_motion_features_df(
    df,
    fps=30.0,
    rom_window_seconds=1.0,
    angles_of_interest=None,
    symmetry_pairs=None,
    frame_column="Frame",
    point_column_prefix="p",
    decimals=6,
):
    """
    Takes a DataFrame with pose landmark columns and returns a new DataFrame
    containing motion features.

    Expected input columns:
        Frame, p11, p12, p13, ...

    Each point cell can be:
        "(x, y, z)"
        or
        (x, y, z)

    Features generated:
        - angle degrees
        - angular velocity
        - vertex joint speed
        - rolling ROM
        - left/right symmetry absolute differences
    """

    if fps <= 0:
        raise ValueError("fps must be greater than 0.")

    if rom_window_seconds <= 0:
        raise ValueError("rom_window_seconds must be greater than 0.")

    if angles_of_interest is None:
        angles_of_interest = DEFAULT_ANGLES_OF_INTEREST

    if symmetry_pairs is None:
        symmetry_pairs = DEFAULT_SYMMETRY_PAIRS

    angle_numbers = make_angle_numbers(angles_of_interest)

    needed_joints = sorted(
        {
            joint
            for angle in angles_of_interest
            for joint in angle
        }
    )

    frames = (
        df[frame_column].astype(float).tolist()
        if frame_column in df.columns
        else list(range(len(df)))
    )

    times = [frame / fps for frame in frames]

    records = []

    for row_index, row in df.iterrows():
        points = {
            joint: parse_point(row[f"{point_column_prefix}{joint}"])
            for joint in needed_joints
        }

        angles = {
            angle: angle_degrees(
                points[angle[0]],
                points[angle[1]],
                points[angle[2]],
            )
            for angle in angles_of_interest
        }

        records.append(
            {
                "frame": frames[row_index],
                "time_sec": times[row_index],
                "points": points,
                "angles": angles,
            }
        )

    output_rows = []

    rom_cache = {}
    for angle in angles_of_interest:
        angle_values = [record["angles"][angle] for record in records]
        rom_cache[angle] = calculate_rolling_rom(
            angle_values,
            times,
            rom_window_seconds,
        )

    for i, record in enumerate(records):
        previous_record = records[i - 1] if i > 0 else None

        if previous_record is None:
            delta_time = None
        else:
            delta_time = record["time_sec"] - previous_record["time_sec"]
            if delta_time <= 0:
                delta_time = None

        output_row = {
            frame_column: record["frame"],
            "time_sec": record["time_sec"],
        }

        # Angles
        for angle in angles_of_interest:
            output_row[angle_column_name(angle, angle_numbers)] = record["angles"][angle]

        # Angular velocities
        for angle in angles_of_interest:
            previous_angle = (
                previous_record["angles"][angle]
                if previous_record is not None
                else None
            )

            output_row[angular_velocity_column_name(angle, angle_numbers)] = (
                calculate_velocity(
                    record["angles"][angle],
                    previous_angle,
                    delta_time,
                )
            )

        # Vertex speeds
        for angle in angles_of_interest:
            vertex_joint = angle[1]

            previous_point = (
                previous_record["points"][vertex_joint]
                if previous_record is not None
                else None
            )

            output_row[vertex_speed_column_name(angle, angle_numbers)] = (
                calculate_speed(
                    record["points"][vertex_joint],
                    previous_point,
                    delta_time,
                )
            )

        # Rolling ROM
        for angle in angles_of_interest:
            output_row[rom_column_name(angle, angle_numbers, rom_window_seconds)] = (
                rom_cache[angle][i]
            )

        # Symmetry
        for left_angle, right_angle in symmetry_pairs:
            left_value = record["angles"].get(left_angle)
            right_value = record["angles"].get(right_angle)

            if left_value is None or right_value is None:
                symmetry = None
            else:
                symmetry = abs(left_value - right_value)

            output_row[symmetry_column_name(left_angle, right_angle, angle_numbers)] = symmetry

        output_rows.append(output_row)

    output_df = pd.DataFrame(output_rows)

    if decimals is not None:
        numeric_columns = output_df.select_dtypes(include="number").columns
        output_df[numeric_columns] = output_df[numeric_columns].round(decimals)

    return output_df