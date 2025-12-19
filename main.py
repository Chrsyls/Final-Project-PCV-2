import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ================= CONFIG =================
WINDOW_NAME = "Kamen Rider – HENSHIN MODE"
BLINK_RATIO_THRESHOLD = 5.0
MOUTH_OPEN_THRESHOLD = 0.5

COLOR_MAIN = (0, 170, 0)
COLOR_DARK = (0, 90, 0)
COLOR_SILVER = (200, 200, 200)
COLOR_EYE = (0, 0, 255)
COLOR_BLACK = (30, 30, 30)

# ================= MEDIAPIPE (FIXED INIT) =================
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=False,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ================= ONE EURO FILTER =================
class OneEuro:
    def __init__(self, freq=30, mincutoff=1.2, beta=0.03):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.prev = None

    def alpha(self, cutoff):
        te = 1 / self.freq
        tau = 1 / (2 * math.pi * cutoff)
        return 1 / (1 + tau / te)

    def filter(self, x):
        if self.prev is None:
            self.prev = x
            return x
        dx = (x - self.prev) * self.freq
        cutoff = self.mincutoff + self.beta * abs(dx)
        a = self.alpha(cutoff)
        self.prev = a * x + (1 - a) * self.prev
        return self.prev

filters = {}
def smooth(name, x, y):
    if name not in filters:
        filters[name] = (OneEuro(), OneEuro())
    fx, fy = filters[name]
    return int(fx.filter(x)), int(fy.filter(y))

# ================= UTILS =================
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def blink_ratio(e, lm):
    return dist(lm[e[0]], lm[e[2]]) / max(1, dist(lm[e[1]], lm[e[3]]))

def mouth_ratio(m, lm):
    return dist(lm[m[1]], lm[m[3]]) / max(1, dist(lm[m[0]], lm[m[2]]))

FINGER_CHAINS = [
    [0,1,2,3,4],
    [0,5,6,7,8],
    [0,9,10,11,12],
    [0,13,14,15,16],
    [0,17,18,19,20]
]

# ================= HAND CACHE =================
hand_cache = {}
HAND_TIMEOUT = 10
frame_id = 0

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    frame_id += 1
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    out = np.zeros_like(frame)

    pose_r = pose.process(rgb)
    face_r = face.process(rgb)
    hands_r = hands.process(rgb)

    blinking = False
    mouth_open = False

    # ===== FACE =====
    if face_r.multi_face_landmarks:
        lm = face_r.multi_face_landmarks[0].landmark
        pts = [(int(p.x*w), int(p.y*h)) for p in lm]
        blinking = (blink_ratio([33,159,133,145], pts) +
                    blink_ratio([362,386,263,374], pts))/2 > BLINK_RATIO_THRESHOLD
        mouth_open = mouth_ratio([61,13,291,14], pts) > MOUTH_OPEN_THRESHOLD

    # ===== BODY =====
    if pose_r.pose_landmarks:
        lm = pose_r.pose_landmarks.landmark
        def P(i, n): return smooth(n, lm[i].x*w, lm[i].y*h)

        nose = P(mp_pose.PoseLandmark.NOSE, "nose")
        l_sh = P(mp_pose.PoseLandmark.LEFT_SHOULDER, "lsh")
        r_sh = P(mp_pose.PoseLandmark.RIGHT_SHOULDER, "rsh")
        l_el = P(mp_pose.PoseLandmark.LEFT_ELBOW, "lel")
        r_el = P(mp_pose.PoseLandmark.RIGHT_ELBOW, "rel")
        l_wr = P(mp_pose.PoseLandmark.LEFT_WRIST, "lwr")
        r_wr = P(mp_pose.PoseLandmark.RIGHT_WRIST, "rwr")
        l_hip = P(mp_pose.PoseLandmark.LEFT_HIP, "lhip")
        r_hip = P(mp_pose.PoseLandmark.RIGHT_HIP, "rhip")
        l_kn = P(mp_pose.PoseLandmark.LEFT_KNEE, "lkn")
        r_kn = P(mp_pose.PoseLandmark.RIGHT_KNEE, "rkn")
        l_an = P(mp_pose.PoseLandmark.LEFT_ANKLE, "lan")
        r_an = P(mp_pose.PoseLandmark.RIGHT_ANKLE, "ran")

        neck = ((l_sh[0]+r_sh[0])//2, (l_sh[1]+r_sh[1])//2)
        pelvis = ((l_hip[0]+r_hip[0])//2, (l_hip[1]+r_hip[1])//2)
        scale = max(0.6, dist(l_sh, r_sh)/190)

        # LEGS
        cv2.line(out, l_hip, l_kn, COLOR_MAIN, int(40*scale))
        cv2.line(out, r_hip, r_kn, COLOR_MAIN, int(40*scale))
        cv2.line(out, l_kn, l_an, COLOR_SILVER, int(28*scale))
        cv2.line(out, r_kn, r_an, COLOR_SILVER, int(28*scale))

        # CHEST
        cv2.line(out, neck, pelvis, COLOR_MAIN, int(22*scale))

        # BELT
        pulse = int(120 + 80*math.sin(time.time()*6))
        cv2.circle(out, pelvis, int(34*scale), (0,0,pulse), -1)

        # ARMS
        cv2.line(out, l_sh, l_el, COLOR_MAIN, int(28*scale))
        cv2.line(out, r_sh, r_el, COLOR_MAIN, int(28*scale))
        cv2.line(out, l_el, l_wr, COLOR_SILVER, int(22*scale))
        cv2.line(out, r_el, r_wr, COLOR_SILVER, int(22*scale))

        # HELMET
        cv2.circle(out, nose, int(60*scale), COLOR_MAIN, -1)
        eye_color = COLOR_BLACK if blinking else COLOR_EYE
        cv2.ellipse(out, (nose[0]-22, nose[1]-8), (22,14), -15, 0, 360, eye_color, -1)
        cv2.ellipse(out, (nose[0]+22, nose[1]-8), (22,14), 15, 0, 360, eye_color, -1)

        mh = int(45*scale if mouth_open else 25*scale)
        cv2.rectangle(out,
            (nose[0]-18, nose[1]+28),
            (nose[0]+18, nose[1]+28+mh),
            COLOR_SILVER, -1
        )

    # ===== HANDS (ANTI-DISAPPEAR) =====
    if hands_r.multi_hand_landmarks:
        for hi, hand in enumerate(hands_r.multi_hand_landmarks):
            pts = [smooth(f"h{hi}_{i}", lm.x*w, lm.y*h)
                   for i, lm in enumerate(hand.landmark)]
            hand_cache[hi] = (pts, frame_id)

    for hi, (pts, last_seen) in list(hand_cache.items()):
        if frame_id - last_seen > HAND_TIMEOUT:
            del hand_cache[hi]
            continue
        for chain in FINGER_CHAINS:
            for i in range(len(chain)-1):
                cv2.line(out, pts[chain[i]], pts[chain[i+1]],
                         COLOR_SILVER, int(7*scale))

    cv2.imshow(WINDOW_NAME, out)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()