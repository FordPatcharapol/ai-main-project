from ultralytics import YOLO
from deepface import DeepFace
from helpers.video import VideoUitls

video_util = VideoUitls()

bgr_green = (0, 255, 0)
bgr_red = (0, 0, 255)

def load_emotion_model(model_path):
    return YOLO(model_path)

def detect_emotion(model, frame):
    image = frame.copy()
    emotion = []

    results = model(image, verbose=False)
    drawing_response = video_util.get_default_display_frame()

    for face in results[0].boxes.data.tolist():
        rectangle_position = []
        text_position = []

        xmin, ymin, xmax, ymax = face[0], face[1], face[2], face[3]
        
        rectangle_position_toappend = {
            "start_point": (int(xmin), int(ymin)),
            "end_point": (int(xmax), int(ymax)),
            "thickness": 2,
        }
        text_position_toappend = {
            "coord":  (int(xmin), int(ymin - 10)),
            "thickness": 2,
        } 

        try:
            analyze = DeepFace.analyze(frame, actions=['emotion'], silent=True)
            emotion.append(analyze[0]['dominant_emotion'])
            
            rectangle_position_toappend["color"] = bgr_green
            text_position_toappend["text"] = str(analyze[0]['dominant_emotion'])
            text_position_toappend["color"] = bgr_green
         
        except:
            rectangle_position_toappend["color"] = bgr_red
            text_position_toappend["text"] = "can't detect"
            text_position_toappend["color"] = bgr_red

        rectangle_position.append(rectangle_position_toappend)
        text_position.append(text_position_toappend)

        drawing_response["rectangle"] += rectangle_position
        drawing_response["text"] += text_position

    return emotion, drawing_response
