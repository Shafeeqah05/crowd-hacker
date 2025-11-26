import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# BASIC PAGE SETTINGS
st.set_page_config(
    page_title="Crowd Detection Admin",
    page_icon="👮",
    layout="wide"
)

# LOAD YOLo
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# SESSION STATE FOR LOGIN
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# LOGIN PAGE
def login_page():
    st.markdown("## 👮 Admin Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

        if login_btn:
            if username == "admin" and password == "admin":
                st.session_state.logged_in = True
                st.success("Login successful! You can now access the Dashboard.")
                st.rerun()
            else:
                st.error("Invalid username or password. Try again.")

# CROWD DETECTION FUNCTIONS
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
        if cls == 0:  # COCO class 0 = 'person'
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

# DASHBOARD PAGE
def dashboard_page():
    st.sidebar.title("👮 Admin Panel")
    st.sidebar.success("Logged in as: admin")

    # Threshold for crowd alert
    threshold = st.sidebar.slider(
        "Crowd Alert Threshold (people)",
        min_value=1,
        max_value=50,
        value=2,
        step=1,
    )

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.title("📊 Crowd Detection Dashboard")
    st.write("Upload an image or video to detect people and measure crowd density.")

    tabs = st.tabs(["🖼 Image Detection", "🎥 Video Detection", "📡 Live Camera"])

    #IMAGE TAB ----------------
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
                st.success(f"Detected **{person_count}** people in this image.")

                if person_count >= threshold:
                    st.error(
                        f"⚠ Crowd Alert! People count ({person_count}) "
                        f"is above threshold ({threshold})."
                    )

                    #  Waiting time block ---
                    extra_people = person_count - threshold
                    minutes_per_person = 2
                    wait_time = max(1, extra_people * minutes_per_person)

                    st.warning(
                        f"⏳ It is crowded now.\n\n"
                        f"Estimated waiting time is about **{wait_time} minutes**.\n"
                        f"Please come after **{wait_time} minutes** for a freer area."
                    )
                   
                else:
                    st.info(
                        f"✅ Safe Crowd Level. People count ({person_count}) "
                        f"is below threshold ({threshold})."
                    )

    #  VIDEO TAB ---
    with tabs[1]:
        st.subheader("Upload a Video")

        video_file = st.file_uploader(
            "Choose a video (short clip recommended)",
            type=["mp4", "avi", "mov"],
            key="video_uploader",
        )

        if video_file is not None:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            tfile.flush()

            # Show the uploaded video
            st.video(tfile.name)

            # Run detection automatically once a file is uploaded
            frame_count = 0
            total_people = 0
            max_people = 0

            cap = cv2.VideoCapture(tfile.name)

            with st.spinner("Processing video, please wait..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Process every 5th frame to be faster
                    if frame_count % 5 != 0:
                        continue

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    _, person_count = detect_people_on_image(frame_rgb)

                    total_people += person_count
                    max_people = max(max_people, person_count)

            cap.release()
            os.unlink(tfile.name)

            if frame_count > 0:
                avg_people = total_people / max(1, frame_count // 5)
            else:
                avg_people = 0

            st.write("---")
            st.subheader("Video Crowd Summary")
            st.write(f"🔢 **Maximum people in a frame:** {int(max_people)}")
            st.write(f"📊 **Average people per processed frame:** {avg_people:.2f}")

            if max_people >= threshold:
                st.error(
                    f"⚠ Crowd Alert! Max people count ({int(max_people)}) "
                    f"is above threshold ({threshold})."
                )
            else:
                st.info(
                    f"✅ Safe Crowd Level. Max people count ({int(max_people)}) "
                    f"is below threshold ({threshold})."
                )

    #   LIVE CAMERA TAB ---
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

                # Draw detections
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crowd alert display
                if person_count >= self.threshold:
                    cv2.putText(
                        img,
                        f"⚠ CROWD ALERT! ({person_count})",
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
if st.session_state.logged_in:
    dashboard_page()
else:

    login_page()
