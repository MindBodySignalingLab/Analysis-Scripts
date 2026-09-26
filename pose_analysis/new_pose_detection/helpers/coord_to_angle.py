import ast
import math
import pandas as pd


DEFAULT_ANGLE_DEFINITIONS = [
    ("angle_11_13_15_at_13", 11, 13, 15),
    ("angle_12_14_16_at_14", 12, 14, 16),
    ("angle_11_13_23_at_13", 11, 13, 23),
    ("angle_12_14_24_at_14", 12, 14, 24),
    ("angle_11_23_25_at_23", 11, 23, 25),
    ("angle_12_24_26_at_24", 12, 24, 26),
]


def parse_point(value):
    """
    Converts '(x, y, z)' string or existing tuple/list into a 3D point tuple.
    """
    if isinstance(value, str):
        value = ast.literal_eval(value)

    return tuple(float(v) for v in value)


def subtract(a, b):
    return (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def angle_degrees(point_a, vertex_b, point_c):
    """
    Return angle ABC in degrees.
    The middle point B is the vertex.
    """
    vector_ba = subtract(point_a, vertex_b)
    vector_bc = subtract(point_c, vertex_b)

    denominator = norm(vector_ba) * norm(vector_bc)

    if denominator == 0:
        return None

    cosine = dot(vector_ba, vector_bc) / denominator
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def calculate_pose_angles(
    df,
    angle_definitions=None,
    frame_column="Frame",
    point_column_prefix="p",
    decimals=6,
    keep_frame=True,
):
    """
    Takes a dataframe containing pose landmark columns and returns
    a new dataframe containing computed angle columns.

    Expected point columns look like:
    p11, p12, p13, etc.

    Each cell can be either:
    - string: "(x, y, z)"
    - tuple/list: (x, y, z)
    """

    if angle_definitions is None:
        angle_definitions = DEFAULT_ANGLE_DEFINITIONS

    output_rows = []

    for _, row in df.iterrows():
        output_row = {}

        if keep_frame and frame_column in df.columns:
            output_row[frame_column] = row[frame_column]

        needed_landmarks = {
            landmark_id
            for _, a, b, c in angle_definitions
            for landmark_id in (a, b, c)
        }

        points = {}
        for landmark_id in needed_landmarks:
            column_name = f"{point_column_prefix}{landmark_id}"
            points[landmark_id] = parse_point(row[column_name])

        for angle_name, a, b, c in angle_definitions:
            value = angle_degrees(points[a], points[b], points[c])

            if value is None:
                output_row[angle_name] = None
            else:
                output_row[angle_name] = round(value, decimals)

        output_rows.append(output_row)

    return pd.DataFrame(output_rows)