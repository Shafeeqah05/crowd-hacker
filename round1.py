import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime, timedelta
import csv

# -------------------------------------------------
# BASIC PAGE SETTINGS
# -------------------------------------------------
st.set_page_config(
    page_title="Crowd Detection Admin",
    page_icon="👮",
    layout="wide"
)

# -------------------------------------------------
# LOAD YOLO MODEL (once, cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# -------------------------------------------------
# SESSION STATE FOR LOGIN + ROLE + PASSWORDS + STUDENT MODE
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "role" not in st.session_state:
    st.session_state.role = None

if "student_mode" not in st.session_state:
    st.session_state.student_mode = False

# default passwords (can be changed via reset)
if "admin_password" not in st.session_state:
    st.session_state.admin_password = "admin"

if "faculty_password" not in st.session_state:
    st.session_state.faculty_password = "faculty"

# reset code (only people who know this can reset passwords)
RESET_CODE = "1234"

# -------------------------------------------------
# SIMPLE LOGGING (for analytics / finals explanation)
# -------------------------------------------------
LOG_FILE = "crowd_logs.csv"

def ensure_log_file():
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "location",
                "mode",          # image / video
                "people_count",
                "avg_people",
                "threshold",
                "crowd_level",
                "wait_time_minutes"
            ])

def log_event(location, mode, people_count, avg_people, threshold, crowd_level, wait_time):
    """Append one detection record to CSV."""
    ensure_log_file()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            location,
            mode,
            people_count,
            f"{avg_people:.2f}" if avg_people is not None else "",
            threshold,
            crowd_level,
            wait_time
        ])

# -------------------------------------------------
# CROWD LEVEL HELPER
# -------------------------------------------------
def get_crowd_level(count, threshold):
    """
    Classify crowd level based on count vs threshold.
    """
    ratio = count / max(1, threshold)
    if ratio < 0.8:
        return "Low"
    elif ratio < 1.5:
        return "Medium"
    else:
        return "High"

# -------------------------------------------------
# LOGIN PAGE (ADMIN / FACULTY) + STUDENT ENTRY
# -------------------------------------------------
def login_page():
    st.markdown("## 👮 Admin / Faculty Login")

    # ---------------- LOGIN FORM ----------------
    with st.form("login_form"):
        username = st.text_input("Username (admin / faculty)")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        # admin login
        if username == "admin" and password == st.session_state.admin_password:
            st.session_state.logged_in = True
            st.session_state.role = "admin"
            st.session_state.student_mode = False
            st.success("Admin login successful!")
            st.rerun()

        # faculty login
        elif username == "faculty" and password == st.session_state.faculty_password:
            st.session_state.logged_in = True
            st.session_state.role = "faculty"
            st.session_state.student_mode = False
            st.success("Faculty login successful!")
            st.rerun()

        else:
            st.error("Invalid username or password.")

    # ---------------- FORGOT / RESET PASSWORD ----------------
    st.markdown("---")
    with st.expander("Forgot / Reset Password"):
        st.info(
            "Use the reset code provided by the organizers to change a password."
        )

        with st.form("reset_form"):
            account_type = st.selectbox("Account to reset", ["admin", "faculty"])
            reset_code_input = st.text_input("Reset code", type="password")
            new_password = st.text_input("New password", type="password")
            confirm_password = st.text_input("Confirm new password", type="password")
            reset_btn = st.form_submit_button("Reset password")

        if reset_btn:
            if reset_code_input != RESET_CODE:
                st.error("Invalid reset code.")
            elif not new_password:
                st.error("New password cannot be empty.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                if account_type == "admin":
                    st.session_state.admin_password = new_password
                else:
                    st.session_state.faculty_password = new_password

                st.success(f"{account_type.capitalize()} password reset successfully!")

    # ---------------- STUDENT VIEW ENTRY (NO LOGIN) ----------------
    st.markdown("---")
    st.markdown("### 👨‍🎓 Student Access (View Only)")
    st.caption("Students can check current crowd status without logging in.")
    if st.button("Continue as Student (View Only)"):
        st.session_state.student_mode = True
        st.session_state.logged_in = False
        st.session_state.role = "student"
        st.rerun()

# -------------------------------------------------
# CROWD DETECTION FUNCTIONS
# -------------------------------------------------
def detect_people_on_image(image: np.ndarray):
    """
    Runs YOLO on an RGB image and returns:
    - processed image with boxes
    - person_count
    """
    results = model(image)[0]
    person_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # class 'person'
            person_count += 1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            label = f"Person {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    return image, person_count

# -------------------------------------------------
# ADMIN / FACULTY DASHBOARD PAGE (FINALE VERSION)
# -------------------------------------------------
def dashboard_page():
    role = st.session_state.get("role", "admin")

    st.sidebar.title("👮 Control Panel")
    st.sidebar.success(f"Logged in as: {role}")

    # --------- MULTI-LOCATION SUPPORT ----------
    location = st.sidebar.selectbox(
        "Select Location",
        ["Canteen", "Admin Block", "IT Lab", "Library", "Auditorium"]
    )

    threshold = st.sidebar.slider(
        f"Crowd Alert Threshold for {location} (people)",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    # minutes needed for 1 extra person to clear (simple prediction model)
    minutes_per_person = st.sidebar.slider(
        "Minutes for 1 person to clear (for prediction)",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.student_mode = False
        st.rerun()

    st.title("📊 Crowd Detection Dashboard (Finale Version)")
    if role == "admin":
        st.caption("You are logged in as Admin (full access).")
    else:
        st.caption("You are logged in as Faculty (crowd monitoring access).")

    st.write(f"Monitoring location: **{location}**")
    st.write("Upload an image or video to detect people and measure crowd density.")

    tabs = st.tabs(["🖼 Image Detection", "🎥 Video Detection", "📡 Live Camera", "📈 Simple Logs"])

    # ---------------- IMAGE TAB ----------------
    with tabs[0]:
        st.subheader("Upload an Image")
        img_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="image_uploader",
        )

        if img_file is not None:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            st.image(rgb_image, caption="Original Image", use_column_width=True)

            if st.button("🔍 Run Crowd Detection on Image"):
                with st.spinner("Running detection..."):
                    processed_img, person_count = detect_people_on_image(
                        rgb_image.copy()
                    )

                st.image(processed_img, caption="Detection Result", use_column_width=True)
                st.success(f"Detected {person_count} people in this image.")

                # Crowd level for image
                crowd_level = get_crowd_level(person_count, threshold)
                st.write(f"**Crowd Level:** {crowd_level}")

                # --------- PREDICTION FEATURE ----------
                if person_count >= threshold:
                    extra_people = person_count - threshold
                    wait_time = max(1, extra_people * minutes_per_person)
                    clear_time = datetime.now() + timedelta(minutes=wait_time)

                    st.error(
                        f"CROWD ALERT! People count ({person_count}) "
                        f"is above threshold ({threshold}) at {location}."
                    )
                    st.warning(
                        f"Estimated waiting time until it becomes free: **{wait_time} minutes**.\n\n"
                        f"Expected to be free around **{clear_time.strftime('%I:%M %p')}**."
                    )
                else:
                    wait_time = 0
                    st.info(
                        f"Safe crowd level. People count ({person_count}) "
                        f"is below threshold ({threshold}) at {location}."
                    )
                    st.success("✅ Area is already free. No waiting time.")

                # Log this detection
                log_event(
                    location=location,
                    mode="image",
                    people_count=person_count,
                    avg_people=None,
                    threshold=threshold,
                    crowd_level=crowd_level,
                    wait_time=wait_time,
                )

    # ---------------- VIDEO TAB ----------------
    with tabs[1]:
        st.subheader("Upload a Video")
        video_file = st.file_uploader(
            "Choose a video (short clip recommended)",
            type=["mp4", "avi", "mov"],
            key="video_uploader",
        )

        if video_file is not None:
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            tfile.flush()

            # Show video preview
            st.video(tfile.name)

            # Only run detection when user clicks the button
            if st.button("🔍 Run Crowd Detection on Video"):
                cap = cv2.VideoCapture(tfile.name)
                frame_count = 0
                total_people = 0
                max_people = 0

                with st.spinner("Processing video..."):
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        # process every 5th frame to speed up
                        if frame_count % 5 != 0:
                            continue

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        _, person_count = detect_people_on_image(frame_rgb)

                        total_people += person_count
                        max_people = max(max_people, person_count)

                cap.release()
                # don't delete the temp file here to avoid Windows PermissionError

                processed_frames = max(1, frame_count // 5)
                avg_people = total_people / processed_frames

                st.write("---")
                st.subheader("Video Crowd Summary")
                st.write(f"Maximum people in a frame: **{int(max_people)}**")
                st.write(f"Average people per processed frame: **{avg_people:.2f}**")

                # Crowd level based on peak crowd
                crowd_level_video = get_crowd_level(max_people, threshold)
                st.write(f"Crowd Level (based on peak crowd): **{crowd_level_video}**")

                # Prediction similar to image, but based on peak crowd
                if max_people >= threshold:
                    extra_people = max_people - threshold
                    wait_time = max(1, extra_people * minutes_per_person)
                    clear_time = datetime.now() + timedelta(minutes=wait_time)

                    st.error(
                        f"CROWD ALERT! Max people count ({int(max_people)}) "
                        f"is above threshold ({threshold}) at {location}."
                    )
                    st.warning(
                        f"Estimated waiting time until it becomes free: **{wait_time} minutes**.\n\n"
                        f"Expected to be free around **{clear_time.strftime('%I:%M %p')}**."
                    )
                else:
                    wait_time = 0
                    st.info(
                        f"Safe crowd level. Max people count ({int(max_people)}) "
                        f"is below threshold ({threshold}) at {location}."
                    )
                    st.success("✅ Area is already free. No waiting time.")

                # Log this detection
                log_event(
                    location=location,
                    mode="video",
                    people_count=int(max_people),
                    avg_people=avg_people,
                    threshold=threshold,
                    crowd_level=crowd_level_video,
                    wait_time=wait_time,
                )

    # ---------------- LIVE CAMERA TAB ----------------
    with tabs[2]:
        st.subheader("Live Crowd Detection (Webcam)")
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.threshold = threshold

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = model(img)[0]

                person_count = 0
                for box in results.boxes:
                    if int(box.cls[0]) == 0:
                        person_count += 1

                for box in results.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if person_count >= self.threshold:
                    cv2.putText(
                        img,
                        f"CROWD ALERT! ({person_count})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                else:
                    cv2.putText(
                        img,
                        f"People: {person_count}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3,
                    )

                return frame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="live",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    # ---------------- SIMPLE LOGS TAB ----------------
    with tabs[3]:
        st.subheader("Detection Logs (Simple View)")
        if os.path.exists(LOG_FILE):
            import pandas as pd
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df.tail(50), use_container_width=True)
        else:
            st.info("No logs yet. Run image or video detection to generate logs.")

# -------------------------------------------------
# STUDENT VIEW PAGE (READS FROM LOGS ONLY)
# -------------------------------------------------
def student_view_page():
    st.title("👨‍🎓 Student Crowd Status")
    st.caption("View-only page. Data comes from the latest detections done by admin/faculty.")

    location = st.selectbox(
        "Select Location",
        ["Canteen", "Admin Block", "IT Lab", "Library", "Auditorium"]
    )

    st.markdown("---")

    if not os.path.exists(LOG_FILE):
        st.info("No crowd data available yet. Please check again later.")
    else:
        import pandas as pd
        df = pd.read_csv(LOG_FILE)

        if df.empty:
            st.info("No crowd data available yet. Please check again later.")
        else:
            loc_df = df[df["location"] == location]
            if loc_df.empty:
                st.info(f"No data available yet for {location}. Please check later.")
            else:
                latest = loc_df.iloc[-1]
                timestamp = latest["timestamp"]
                people_count = latest["people_count"]
                crowd_level = latest["crowd_level"]
                mode = latest["mode"]
                wait_raw = latest.get("wait_time_minutes", "")

                try:
                    wait_time = float(wait_raw) if wait_raw != "" else 0
                except Exception:
                    wait_time = 0

                st.subheader(f"Latest Status for {location}")
                st.write(f"Last updated: **{timestamp}**")
                st.write(f"Detection source: **{str(mode).capitalize()}**")

                # Display crowd level with color
                if str(crowd_level).lower() == "low":
                    st.success(f"Crowd Level: {crowd_level}")
                elif str(crowd_level).lower() == "medium":
                    st.warning(f"Crowd Level: {crowd_level}")
                else:
                    st.error(f"Crowd Level: {crowd_level}")

                st.write(f"Detected people: **{people_count}**")

                if wait_time and wait_time > 0:
                    st.warning(f"Estimated waiting time: **{int(wait_time)} minutes**")
                else:
                    st.success("Area is currently free. No waiting time expected.")

                st.markdown("### Recent records for this location")
                st.dataframe(loc_df.tail(10), use_container_width=True)

    st.markdown("---")
    if st.button("🔑 Go to Admin / Faculty Login"):
        st.session_state.student_mode = False
        st.rerun()

# -------------------------------------------------
# ROUTING
# -------------------------------------------------
if st.session_state.get("student_mode", False):
    # Student view (no login required)
    student_view_page()
elif st.session_state.logged_in:
    # Admin / Faculty dashboard
    dashboard_page()
else:
    # Login page
    login_page()


