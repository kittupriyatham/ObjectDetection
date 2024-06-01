from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            face_detector = cv2.CascadeClassifier('static/OpenCV XML files/haarcascades/haarcascade_frontalface_default.xml')
            eye_detector = cv2.CascadeClassifier('static/OpenCV XML files/haarcascades/haarcascade_eye.xml')
            smile_detector = cv2.CascadeClassifier('static/OpenCV XML files/haarcascades/haarcascade_smile.xml')
            faces = face_detector.detectMultiScale(frame, 1.3, 5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                eyes = eye_detector.detectMultiScale(roi_gray, 1.1, 3)
                smiles = smile_detector.detectMultiScale(roi_gray, 1.8, 20)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
