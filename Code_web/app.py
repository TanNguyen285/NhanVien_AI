
import os
import time

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for

import inference
import scan as scan_mod
from camera import Camera

# ---------------------------------------------------------------------------
# CAMERA
# ---------------------------------------------------------------------------
cam = Camera(src=0, reconnect_interval=10.0, max_fail=20)
cam.start()

# ---------------------------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key')

EMPLOYEES = {
    'alice': '1',
    'bob':   '2',
}

# ---------------------------------------------------------------------------
# MJPEG STREAM
# ---------------------------------------------------------------------------
def _gen_frames():
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        inference.push_frame(frame)   # pipeline song song

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.001)


# ---------------------------------------------------------------------------
# ROUTES — Auth
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if not session.get('user'):
        return redirect(url_for('login'))
    return render_template("index.html", user=session.get('user'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if EMPLOYEES.get(username) == password:
            session['user'] = username
            return redirect(url_for('index'))
        return render_template('login.html', error='Sai tên đăng nhập hoặc mật khẩu')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


# ---------------------------------------------------------------------------
# ROUTES — Stream
# ---------------------------------------------------------------------------
@app.route('/video_feed')
def video_feed():
    if not session.get('user'):
        return '', 401
    return Response(_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# ROUTES — Predict
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["GET"])
def predict_emotion():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"results": inference.get_latest()})


# ---------------------------------------------------------------------------
# ROUTES — Scan
# ---------------------------------------------------------------------------
@app.route("/scan/start", methods=["POST"])
def scan_start():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    result = scan_mod.start(cam)
    code = 409 if result["status"] == "already_running" else 200
    return jsonify(result), code


@app.route("/scan/status", methods=["GET"])
def scan_status():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(scan_mod.status())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        host_ip = os.popen("hostname -I").read().split()[0]
    except Exception:
        host_ip = "0.0.0.0"
    print(f"Dashboard: http://{host_ip}:8000")
    try:
        app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
    finally:
        inference.stop()
        cam.stop()