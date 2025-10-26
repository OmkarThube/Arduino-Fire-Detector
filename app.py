from flask import Flask, render_template, Response, request, redirect
import cv2
import numpy as np

app = Flask(__name__)

# Define default color ranges
color_ranges = {
    'Red': ([0, 120, 70], [10, 255, 255]),
    'Red2': ([170, 120, 70], [180, 255, 255]),
    'Green': ([36, 25, 25], [86, 255, 255]),
    'Blue': ([94, 80, 2], [126, 255, 255]),
    'Orange': ([10, 100, 20], [25, 255, 255]),
    'Purple': ([130, 50, 50], [160, 255, 255]),
    'Yellow': ([20, 100, 100], [30, 255, 255]),
    'Pink': ([145, 100, 100], [165, 255, 255]),
    'Cyan': ([85, 100, 100], [95, 255, 255]),
    'Brown': ([10, 100, 20], [20, 255, 200]),
    'Gray': ([0, 0, 50], [180, 25, 200]),
    'Black': ([0, 0, 0], [180, 255, 30]),
    'White': ([0, 0, 200], [180, 20, 255]),
}

# For tracking objects
tracked_objects = {}

def detect_colors(frame):
    # Histogram equalization for robustness to lighting changes
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame[..., 0] = cv2.equalizeHist(frame[..., 0])  # Equalize Y channel
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_objects = []

    for color_name, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust threshold
                x, y, w, h = cv2.boundingRect(contour)
                detected_objects.append((color_name, (x, y, w, h)))

    # Track detected objects using centroid
    return track_objects(detected_objects)

def track_objects(detected_objects):
    global tracked_objects
    new_tracked_objects = {}

    for color_name, (x, y, w, h) in detected_objects:
        centroid = (int(x + w / 2), int(y + h / 2))
        if color_name not in tracked_objects:
            tracked_objects[color_name] = centroid
            new_tracked_objects[color_name] = (x, y, w, h)
        else:
            prev_centroid = tracked_objects[color_name]
            # Update position with some thresholding for tracking
            if np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) < 50:
                new_tracked_objects[color_name] = (x, y, w, h)
                tracked_objects[color_name] = centroid  # Update tracked position

    # Update tracked objects
    for color_name in list(tracked_objects.keys()):
        if color_name not in new_tracked_objects:
            del tracked_objects[color_name]

    return new_tracked_objects

def generate_frames():
    cap = cv2.VideoCapture(0)  # Start video capture
    frame_skip = 0  # Frame skip counter

    while True:
        success, frame = cap.read()  # Read frame from webcam
        if not success:
            break

        # Skip frames for performance
        frame_skip += 1
        if frame_skip % 2 != 0:  # Process every second frame
            continue

        # Process the frame for color detection
        detected_objects = detect_colors(frame)

        # Draw bounding boxes and display detected color
        for color_name, (x, y, w, h) in detected_objects.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for rendering in the HTML
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the capture

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' in request.files:
        # Get the uploaded file
        file = request.files['file']
        if file:
            file_path = "uploaded_image.jpg"
            file.save(file_path)
            image = cv2.imread(file_path)

            # Process the image for color detection
            detected_colors = detect_colors(image)

            # Return the results page with the detected colors
            return render_template('results.html', colors=detected_colors)

    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)
