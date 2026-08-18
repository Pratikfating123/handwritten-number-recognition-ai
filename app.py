import os
import io
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    jsonify,
    render_template,
    request
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "digit_model.keras"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "predictions.db"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully!")

except Exception as e:

    print("ERROR: Could not load model.")
    print(e)

    model = None


# ============================================================
# DATABASE
# ============================================================

def get_db():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():

    connection = get_db()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            number TEXT NOT NULL,

            confidence REAL NOT NULL,

            digit_count INTEGER NOT NULL,

            created_at TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()


init_database()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# IMAGE READING
# ============================================================

def read_uploaded_image(file):

    image_bytes = file.read()

    if not image_bytes:

        raise ValueError(
            "The uploaded image is empty."
        )


    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8
    )


    image = cv2.imdecode(
        array,
        cv2.IMREAD_GRAYSCALE
    )


    if image is None:

        raise ValueError(
            "Could not read the uploaded image."
        )


    return image


# ============================================================
# CLEAN CANVAS
# ============================================================

def prepare_canvas(image):

    # Resize to a predictable working size

    image = cv2.resize(
        image,
        (280, 280),
        interpolation=cv2.INTER_AREA
    )


    # Gaussian blur removes tiny noise

    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )


    # Convert black handwriting to white foreground

    _, binary = cv2.threshold(
        blurred,
        180,
        255,
        cv2.THRESH_BINARY_INV
    )


    # Remove tiny noise

    kernel = np.ones(
        (3, 3),
        np.uint8
    )


    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )


    # Slightly connect broken strokes

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )


    return binary


# ============================================================
# FIND DIGIT CONTOURS
# ============================================================

def find_digit_regions(binary):

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    regions = []


    canvas_area = (
        binary.shape[0] *
        binary.shape[1]
    )


    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )


        area = cv2.contourArea(
            contour
        )


        # Ignore extremely tiny objects

        if area < 25:
            continue


        # Ignore objects that occupy
        # almost the entire canvas

        if (
            w * h
        ) > canvas_area * 0.80:

            continue


        # Ignore very small noise

        if w < 8 or h < 12:
            continue


        regions.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area
            }
        )


    # Sort from left to right

    regions.sort(
        key=lambda item: item["x"]
    )


    return regions


# ============================================================
# MERGE FRAGMENTED DIGITS
# ============================================================

def merge_close_regions(
    regions
):

    if len(regions) <= 1:

        return regions


    merged = []

    current = regions[0].copy()


    for next_region in regions[1:]:

        current_right = (
            current["x"] +
            current["w"]
        )


        gap = (
            next_region["x"] -
            current_right
        )


        current_height = current["h"]

        next_height = next_region["h"]


        height_ratio = (
            min(
                current_height,
                next_height
            )
            /
            max(
                current_height,
                next_height
            )
        )


        # If two pieces are very close
        # and have similar height, merge them.

        if (
            gap < 8
            and height_ratio > 0.45
        ):

            new_x = min(
                current["x"],
                next_region["x"]
            )

            new_y = min(
                current["y"],
                next_region["y"]
            )

            new_right = max(
                current["x"] +
                current["w"],

                next_region["x"] +
                next_region["w"]
            )

            new_bottom = max(
                current["y"] +
                current["h"],

                next_region["y"] +
                next_region["h"]
            )


            current = {

                "x": new_x,

                "y": new_y,

                "w": new_right - new_x,

                "h": new_bottom - new_y,

                "area":
                    current["area"] +
                    next_region["area"]

            }

        else:

            merged.append(
                current
            )

            current = (
                next_region.copy()
            )


    merged.append(
        current
    )


    return merged


# ============================================================
# SPLIT VERY WIDE REGIONS
# ============================================================

def split_wide_regions(
    binary,
    regions
):

    result = []


    for region in regions:

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]


        # Normal digit

        if w <= h * 1.35:

            result.append(
                region
            )

            continue


        # Wide handwriting may contain
        # two touching digits.

        crop = binary[
            y:y + h,
            x:x + w
        ]


        vertical_projection = np.sum(
            crop > 0,
            axis=0
        )


        # Search for low-density valleys

        threshold = max(
            2,
            int(h * 0.04)
        )


        valleys = []


        for i in range(
            8,
            len(vertical_projection) - 8
        ):

            local = vertical_projection[
                i - 4:i + 5
            ]


            if (
                vertical_projection[i]
                <= threshold
                and
                vertical_projection[i]
                ==
                np.min(local)
            ):

                valleys.append(i)


        if not valleys:

            result.append(
                region
            )

            continue


        # Pick a valley close to the center

        center = w / 2

        split_x = min(
            valleys,
            key=lambda value:
            abs(value - center)
        )


        # Don't create extremely
        # small regions.

        if (
            split_x < w * 0.25
            or split_x > w * 0.75
        ):

            result.append(
                region
            )

            continue


        left = {

            "x": x,

            "y": y,

            "w": split_x,

            "h": h,

            "area":
                int(
                    np.sum(
                        crop[:, :split_x]
                        > 0
                    )
                )

        }


        right = {

            "x": x + split_x,

            "y": y,

            "w": w - split_x,

            "h": h,

            "area":
                int(
                    np.sum(
                        crop[:, split_x:]
                        > 0
                    )
                )

        }


        if (
            left["w"] >= 10
            and right["w"] >= 10
        ):

            result.append(
                left
            )

            result.append(
                right
            )

        else:

            result.append(
                region
            )


    result.sort(
        key=lambda item: item["x"]
    )


    return result


# ============================================================
# MNIST STYLE PREPROCESSING
# ============================================================

def make_mnist_image(
    digit_crop
):

    # Make sure foreground is binary

    _, digit_crop = cv2.threshold(
        digit_crop,
        80,
        255,
        cv2.THRESH_BINARY
    )


    # Find foreground pixels

    points = cv2.findNonZero(
        digit_crop
    )


    if points is None:

        raise ValueError(
            "No digit detected."
        )


    # Tight crop

    x, y, w, h = cv2.boundingRect(
        points
    )


    digit_crop = digit_crop[
        y:y + h,
        x:x + w
    ]


    # Add padding around digit

    padding = 4


    padded = cv2.copyMakeBorder(

        digit_crop,

        padding,
        padding,
        padding,
        padding,

        cv2.BORDER_CONSTANT,

        value=0

    )


    h, w = padded.shape


    # MNIST normally occupies around
    # 20x20 pixels inside a 28x28 image.

    target_size = 20


    scale = min(
        target_size / w,
        target_size / h
    )


    new_w = max(
        1,
        int(round(w * scale))
    )


    new_h = max(
        1,
        int(round(h * scale))
    )


    resized = cv2.resize(

        padded,

        (
            new_w,
            new_h
        ),

        interpolation=cv2.INTER_AREA

    )


    # Create 28x28 black canvas

    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # CENTER DIGIT
    # --------------------------------------------------------

    start_x = (
        28 - new_w
    ) // 2


    start_y = (
        28 - new_h
    ) // 2


    canvas[
        start_y:start_y + new_h,
        start_x:start_x + new_w
    ] = resized


    # --------------------------------------------------------
    # CENTER OF MASS CORRECTION
    # --------------------------------------------------------

    moments = cv2.moments(
        canvas
    )


    if moments["m00"] != 0:

        center_x = (
            moments["m10"] /
            moments["m00"]
        )


        center_y = (
            moments["m01"] /
            moments["m00"]
        )


        shift_x = int(
            round(
                13.5 - center_x
            )
        )


        shift_y = int(
            round(
                13.5 - center_y
            )
        )


        matrix = np.float32(
            [
                [1, 0, shift_x],
                [0, 1, shift_y]
            ]
        )


        canvas = cv2.warpAffine(

            canvas,

            matrix,

            (28, 28),

            borderMode=cv2.BORDER_CONSTANT,

            borderValue=0

        )


    # Slight blur gives a more
    # MNIST-like smooth stroke.

    canvas = cv2.GaussianBlur(
        canvas,
        (3, 3),
        0
    )


    return canvas


# ============================================================
# CREATE PREDICTION VARIANTS
# ============================================================

def create_prediction_variants(
    mnist_image
):

    variants = []


    # Variant 1: normal

    variants.append(
        mnist_image
    )


    # Variant 2: slightly thicker

    kernel = np.ones(
        (2, 2),
        np.uint8
    )


    thicker = cv2.dilate(
        mnist_image,
        kernel,
        iterations=1
    )


    variants.append(
        thicker
    )


    # Variant 3: slightly thinner

    thinner = cv2.erode(
        mnist_image,
        kernel,
        iterations=1
    )


    variants.append(
        thinner
    )


    return variants


# ============================================================
# PREDICT SINGLE DIGIT
# ============================================================

def predict_single_digit(
    digit_image
):

    variants = (
        create_prediction_variants(
            digit_image
        )
    )


    predictions = []


    for variant in variants:

        normalized = (
            variant.astype(
                "float32"
            ) / 255.0
        )


        normalized = np.expand_dims(
            normalized,
            axis=0
        )


        normalized = np.expand_dims(
            normalized,
            axis=-1
        )


        prediction = model.predict(
            normalized,
            verbose=0
        )[0]


        predictions.append(
            prediction
        )


    # Average all preprocessing variants

    probabilities = np.mean(
        predictions,
        axis=0
    )


    predicted_digit = int(
        np.argmax(
            probabilities
        )
    )


    confidence = float(
        probabilities[
            predicted_digit
        ] * 100
    )


    probability_dict = {

        str(i):
            round(
                float(
                    probabilities[i]
                    * 100
                ),
                4
            )

        for i in range(10)

    }


    return (
        predicted_digit,
        confidence,
        probability_dict
    )


# ============================================================
# PREDICT NUMBER
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if model is None:

            return jsonify({

                "success": False,

                "error":
                    "AI model is not loaded.",

                "type":
                    "server"

            }), 500


        # ----------------------------------------------------
        # GET IMAGE
        # ----------------------------------------------------

        file = request.files.get(
            "image"
        )


        if file is None:

            return jsonify({

                "success": False,

                "error":
                    "No drawing was received.",

                "type":
                    "empty"

            }), 400


        image = read_uploaded_image(
            file
        )


        # ----------------------------------------------------
        # PREPARE CANVAS
        # ----------------------------------------------------

        binary = prepare_canvas(
            image
        )


        # ----------------------------------------------------
        # FIND DIGITS
        # ----------------------------------------------------

        regions = find_digit_regions(
            binary
        )


        regions = merge_close_regions(
            regions
        )


        regions = split_wide_regions(
            binary,
            regions
        )


        # ----------------------------------------------------
        # NO DIGIT
        # ----------------------------------------------------

        if not regions:

            return jsonify({

                "success": False,

                "error":
                    "No handwritten digit detected. Please draw a number.",

                "type":
                    "empty"

            }), 400


        # ----------------------------------------------------
        # LIMIT NUMBER OF DIGITS
        # ----------------------------------------------------

        if len(regions) > 10:

            return jsonify({

                "success": False,

                "error":
                    "Too many digits detected. Please draw up to 10 digits.",

                "type":
                    "invalid"

            }), 400


        # ----------------------------------------------------
        # PREDICT EACH DIGIT
        # ----------------------------------------------------

        digits = []

        recognized_number = ""


        for index, region in enumerate(
            regions
        ):

            x = region["x"]
            y = region["y"]
            w = region["w"]
            h = region["h"]


            # Extra padding

            padding = 6


            x1 = max(
                0,
                x - padding
            )


            y1 = max(
                0,
                y - padding
            )


            x2 = min(
                binary.shape[1],
                x + w + padding
            )


            y2 = min(
                binary.shape[0],
                y + h + padding
            )


            crop = binary[
                y1:y2,
                x1:x2
            ]


            # Convert crop into MNIST style

            mnist_digit = make_mnist_image(
                crop
            )


            (
                predicted_digit,
                confidence,
                probabilities
            ) = predict_single_digit(
                mnist_digit
            )


            recognized_number += str(
                predicted_digit
            )


            digits.append({

                "position":
                    index + 1,

                "digit":
                    predicted_digit,

                "confidence":
                    round(
                        confidence,
                        2
                    ),

                "probabilities":
                    probabilities,

                "box": {

                    "x": int(x),

                    "y": int(y),

                    "width": int(w),

                    "height": int(h)

                }

            })


        # ----------------------------------------------------
        # OVERALL CONFIDENCE
        # ----------------------------------------------------

        individual_confidences = [

            item["confidence"]

            for item in digits

        ]


        if individual_confidences:

            overall_confidence = float(
                np.mean(
                    individual_confidences
                )
            )

        else:

            overall_confidence = 0.0


        overall_confidence = round(
            overall_confidence,
            2
        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        connection = get_db()

        cursor = connection.cursor()


        created_at = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        cursor.execute(

            """
            INSERT INTO predictions
            (
                number,
                confidence,
                digit_count,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,

            (
                recognized_number,
                overall_confidence,
                len(digits),
                created_at
            )

        )


        connection.commit()

        connection.close()


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success":
                True,

            "number":
                recognized_number,

            "digit_count":
                len(digits),

            "overall_confidence":
                overall_confidence,

            "digits":
                digits,

            "message":
                "Number recognized successfully."

        })


    except Exception as e:

        print(
            "\nPrediction error:",
            str(e)
        )


        return jsonify({

            "success":
                False,

            "error":
                "Prediction failed: " +
                str(e),

            "type":
                "server"

        }), 500


# ============================================================
# HISTORY
# ============================================================

@app.route(
    "/history",
    methods=["GET"]
)
def history():

    try:

        connection = get_db()

        cursor = connection.cursor()


        cursor.execute(
            """
            SELECT
                id,
                number,
                confidence,
                digit_count,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 50
            """
        )


        rows = cursor.fetchall()


        cursor.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(
                    AVG(confidence),
                    0
                ) AS average_confidence
            FROM predictions
            """
        )


        stats = cursor.fetchone()


        connection.close()


        history_data = []


        for row in rows:

            history_data.append({

                "id":
                    row["id"],

                "digit":
                    row["number"],

                "confidence":
                    round(
                        float(
                            row["confidence"]
                        ),
                        2
                    ),

                "digit_count":
                    row["digit_count"],

                "time":
                    row["created_at"]

            })


        return jsonify({

            "success":
                True,

            "history":
                history_data,

            "total":
                int(
                    stats["total"]
                    or 0
                ),

            "average_confidence":
                round(
                    float(
                        stats[
                            "average_confidence"
                        ]
                        or 0
                    ),
                    2
                )

        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/history/clear",
    methods=["POST"]
)
def clear_history():

    try:

        connection = get_db()

        cursor = connection.cursor()


        cursor.execute(
            "DELETE FROM predictions"
        )


        connection.commit()

        connection.close()


        return jsonify({

            "success":
                True,

            "message":
                "Prediction history cleared."

        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# ANALYTICS
# ============================================================

@app.route(
    "/analytics",
    methods=["GET"]
)
def analytics():

    try:

        connection = get_db()

        cursor = connection.cursor()


        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM predictions
            """
        )


        total = cursor.fetchone()["total"]


        # ----------------------------------------------------
        # DIGIT COUNTS
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT number
            FROM predictions
            """
        )


        rows = cursor.fetchall()


        digit_counts = {
            str(i): 0
            for i in range(10)
        }


        for row in rows:

            number = str(
                row["number"]
            )


            for digit in number:

                if digit in digit_counts:

                    digit_counts[digit] += 1


        # ----------------------------------------------------
        # MOST COMMON
        # ----------------------------------------------------

        most_common_digit = None

        most_common_count = 0


        if rows:

            most_common_digit = max(
                digit_counts,
                key=digit_counts.get
            )


            most_common_count = (
                digit_counts[
                    most_common_digit
                ]
            )


        # ----------------------------------------------------
        # HIGHEST CONFIDENCE
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                number,
                confidence
            FROM predictions
            ORDER BY confidence DESC
            LIMIT 1
            """
        )


        highest = cursor.fetchone()


        highest_confidence = 0

        highest_confidence_digit = None


        if highest:

            highest_confidence = round(
                float(
                    highest["confidence"]
                ),
                2
            )


            highest_confidence_digit = (
                highest["number"]
            )


        # ----------------------------------------------------
        # LATEST
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                number,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 1
            """
        )


        latest = cursor.fetchone()


        latest_digit = None

        latest_time = None


        if latest:

            latest_digit = (
                latest["number"]
            )

            latest_time = (
                latest["created_at"]
            )


        connection.close()


        return jsonify({

            "success":
                True,

            "total":
                total,

            "digit_counts":
                digit_counts,

            "most_common_digit":
                most_common_digit,

            "most_common_count":
                most_common_count,

            "highest_confidence":
                highest_confidence,

            "highest_confidence_digit":
                highest_confidence_digit,

            "latest_digit":
                latest_digit,

            "latest_time":
                latest_time

        })


    except Exception as e:

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "online",

        "model":
            "loaded"
            if model is not None
            else "not loaded",

        "tensorflow":
            tf.__version__

    })


# ============================================================
# SECURITY API
# ============================================================

@app.route(
    "/api/security",
    methods=["GET"]
)
def security_api():

    return jsonify({

        "success":
            True,

        "status":
            "secure",

        "model_loaded":
            model is not None,

        "tensorflow":
            tf.__version__

    })


# ============================================================
# INTERFACES API
# ============================================================

@app.route(
    "/api/interfaces",
    methods=["GET"]
)
def interfaces_api():

    return jsonify({

        "success":
            True,

        "application":
            "Handwritten Number Recognition AI",

        "model":
            "CNN",

        "input":
            "28x28 grayscale",

        "classes":
            10

    })


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "success":
            False,

        "error":
            "Endpoint not found."

    }), 404


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "HANDWRITTEN NUMBER RECOGNITION AI"
    )

    print("=" * 60)

    print(
        "CNN Model      : Loaded"
        if model is not None
        else "CNN Model      : ERROR"
    )

    print(
        "Multi-digit    : Enabled"
    )

    print(
        "Preprocessing  : MNIST optimized"
    )

    print(
        "History        : SQLite enabled"
    )

    print(
        "Server         : http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )