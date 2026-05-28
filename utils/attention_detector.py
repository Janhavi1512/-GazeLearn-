import cv2
from mediapipe.python import solutions as mp
import numpy as np
from scipy.spatial import distance as dist

# ── MediaPipe landmark indices ────────────────────────────────
# Left eye
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398]

# Right eye
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
             133, 173, 157, 158, 159, 160, 161, 246]

# EAR landmark indices
LEFT_EAR_PTS = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_PTS = [33, 160, 158, 133, 153, 144]

# Iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


class AttentionDetector:

    def __init__(self, ear_thresh=0.22, consec_frames=20, gaze_thresh=0.5):

        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.gaze_thresh = gaze_thresh

        self.drowsy_counter = 0

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # ──────────────────────────────────────────────────────────
    def update_thresholds(self, ear_thresh, consec_frames, gaze_thresh):

        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.gaze_thresh = gaze_thresh

    # ──────────────────────────────────────────────────────────
    def process_frame(self, frame):

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        output = {
            "state": "unknown",
            "ear": 0.0,
            "gaze_ratio": 0.0,
            "attention_score": 0,
            "annotated_frame": frame.copy()
        }

        if not results.multi_face_landmarks:

            self._draw_no_face(output["annotated_frame"])

            return output

        landmarks = results.multi_face_landmarks[0].landmark

        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in landmarks],
            dtype=np.float32
        )

        # EAR
        left_ear = self._eye_aspect_ratio(pts, LEFT_EAR_PTS)

        right_ear = self._eye_aspect_ratio(pts, RIGHT_EAR_PTS)

        ear = (left_ear + right_ear) / 2.0

        # Gaze
        gaze_ratio = self._gaze_ratio(pts)

        # Head pose
        head_turned = self._head_pose_check(pts)

        # State classification
        if ear < self.ear_thresh:
            self.drowsy_counter += 1
        else:
            self.drowsy_counter = max(0, self.drowsy_counter - 1)

        if self.drowsy_counter >= self.consec_frames:
            state = "drowsy"

        elif gaze_ratio > self.gaze_thresh or head_turned:
            state = "distracted"

        else:
            state = "focused"

        # Attention score
        attention_score = self._compute_score(state, ear, gaze_ratio)

        # Annotated frame
        annotated = self._annotate(
            frame.copy(),
            pts,
            state,
            ear,
            gaze_ratio,
            attention_score
        )

        output.update({
            "state": state,
            "ear": round(ear, 3),
            "gaze_ratio": round(gaze_ratio, 3),
            "attention_score": attention_score,
            "annotated_frame": annotated
        })

        return output

    # ──────────────────────────────────────────────────────────
    def _eye_aspect_ratio(self, pts, indices):

        p = [pts[i] for i in indices]

        A = dist.euclidean(p[1], p[5])

        B = dist.euclidean(p[2], p[4])

        C = dist.euclidean(p[0], p[3])

        ear = (A + B) / (2.0 * C + 1e-6)

        return ear

    # ──────────────────────────────────────────────────────────
    def _gaze_ratio(self, pts):

        try:
            # Left iris
            l_iris = np.mean([pts[i] for i in LEFT_IRIS], axis=0)

            l_left = pts[LEFT_EAR_PTS[0]]

            l_right = pts[LEFT_EAR_PTS[3]]

            l_width = dist.euclidean(l_left, l_right) + 1e-6

            l_ratio = (l_iris[0] - l_left[0]) / l_width

            # Right iris
            r_iris = np.mean([pts[i] for i in RIGHT_IRIS], axis=0)

            r_left = pts[RIGHT_EAR_PTS[0]]

            r_right = pts[RIGHT_EAR_PTS[3]]

            r_width = dist.euclidean(r_left, r_right) + 1e-6

            r_ratio = (r_iris[0] - r_left[0]) / r_width

            avg = (l_ratio + r_ratio) / 2.0

            deviation = abs(avg - 0.5)

            return round(deviation, 3)

        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────
    def _head_pose_check(self, pts):

        try:
            nose_x = pts[1][0]

            face_left = pts[234][0]

            face_right = pts[454][0]

            face_center = (face_left + face_right) / 2.0

            face_width = abs(face_right - face_left) + 1e-6

            offset = abs(nose_x - face_center) / face_width

            return offset > 0.15

        except Exception:
            return False

    # ──────────────────────────────────────────────────────────
    def _compute_score(self, state, ear, gaze):

        if state == "focused":

            base = 85

            bonus = min(15, int((ear - self.ear_thresh) * 100))

            return min(100, base + bonus)

        elif state == "distracted":

            penalty = int(gaze * 60)

            return max(20, 60 - penalty)

        else:
            return max(0, int(ear * 100))

    # ──────────────────────────────────────────────────────────
    def _annotate(self, frame, pts, state, ear, gaze, score):

        color_map = {
            "focused": (0, 255, 136),
            "distracted": (0, 136, 255),
            "drowsy": (0, 60, 255),
            "unknown": (128, 128, 128)
        }

        color = color_map.get(state, (128, 128, 128))

        # Draw eye points
        for idx in LEFT_EAR_PTS + RIGHT_EAR_PTS:

            pt = (int(pts[idx][0]), int(pts[idx][1]))

            cv2.circle(frame, pt, 2, color, -1)

        h, w = frame.shape[:2]

        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)

        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        label = f"{state.upper()} | EAR:{ear:.2f} | GAZE:{gaze:.2f} | ATT:{score}%"

        cv2.putText(
            frame,
            label,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)

        return frame

    # ──────────────────────────────────────────────────────────
    def _draw_no_face(self, frame):

        h, w = frame.shape[:2]

        cv2.putText(
            frame,
            "No face detected",
            (w // 2 - 130, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (80, 80, 80),
            2,
            cv2.LINE_AA
        )

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (60, 60, 60), 3)