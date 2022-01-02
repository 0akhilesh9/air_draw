
import cv2
import os
import math
import numpy as np
import mediapipe as mp
from scipy.spatial import ConvexHull, convex_hull_plot_2d

import gesture as gesture
import tmp as tmp

test_img_dir = r"D:\cevit\test_images"

images = [cv2.imread(os.path.join(test_img_dir, file_path)) for file_path in os.listdir(test_img_dir)]


"""All MediaPipe Solutions Python API examples are under mp.solutions.

For the MediaPipe Hands solution, we can access this module as `mp_hands = mp.solutions.hands`.

You may change the parameters, such as `static_image_mode`, `max_num_hands`, and `min_detection_confidence`, during the initialization. Run `help(mp_hands.Hands)` to get more informations about the parameters.
"""

mp_hands = mp.solutions.hands
# help(mp_hands.Hands)

mp_drawing = mp.solutions.drawing_utils 
vid = cv2.VideoCapture(0)
frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = vid.get(cv2.CAP_PROP_FPS)
marker_thickness = -1
thickness = 2
selected_radius = 3
drawing_list = []
selected_color = [0, 0, 255]
num_frames = 0
buffer_pixels = 40
bkg_img = None
hand_dist_threshold = 0.2
draw_flag = False
rect_l_t = (frame_width-50,0)
rect_r_b = (frame_width,20)

middle = [0,0]
index = [0,0]
ring = [0,0]
thumb = [0,0]
pinky = [0,0]

def check_color_picker():
    color_picker_bound = 30
    if (abs(thumb[0] - middle[0]) < color_picker_bound) and (abs(thumb[1] - middle[1]) < color_picker_bound) and \
        (abs(thumb[0] - ring[0]) < color_picker_bound*1.5) and (abs(thumb[1] - ring[1]) < color_picker_bound*1.5) and \
        (abs(thumb[0] - pinky[0]) < color_picker_bound*1.5) and (abs(thumb[1] - pinky[1]) < color_picker_bound*1.5):
        return True
    return False

def check_validity(results):
    if (not results.multi_hand_landmarks):
        return False
    elif (len(results.multi_handedness) < 2):
        return False
    else:
        hand1_index_x = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        hand2_index_x = results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        if abs(hand1_index_x - hand2_index_x) < hand_dist_threshold:
            return False
        return True

    results.multi_hand_landmarks

def get_points(hand_landmarks):
    points = []
    for landmark in hand_landmarks.landmark:
        index_x = int(landmark.x * frame_width)
        index_x = max(0, min(index_x, frame_width))
        index_y = int(landmark.y * frame_height)
        index_y = max(0, min(index_y, frame_height))
        points.append([index_x, index_y])
    return points

def get_convex_hull_area(hand_landmarks):
    points = get_points(hand_landmarks)
    hull = ConvexHull(points)
    print(hull.area)

def calculate_points(hand_landmarks):
    global index
    global middle
    global ring
    global thumb
    global pinky
    middle[0] = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame_width)
    middle[0] = max(0, min(middle[0], frame_width))
    middle[1] = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame_height)
    middle[1] = max(0, min(middle[1], frame_height))

    ring[0] = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * frame_width)
    ring[0] = max(0, min(ring[0], frame_width))
    ring[1] = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * frame_height)
    ring[1] = max(0, min(ring[1], frame_height))

    thumb[0] = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame_width)
    thumb[0] = max(0, min(thumb[0], frame_width))
    thumb[1] = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame_height)
    thumb[1] = max(0, min(thumb[1], frame_height))

    index[0] = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width)
    index[0] = max(0, min(index[0], frame_width))
    index[1] = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height)
    index[1] = max(0, min(index[1], frame_height))

    pinky[0] = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * frame_width)
    pinky[0] = max(0, min(pinky[0], frame_width))
    pinky[1] = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * frame_height)
    pinky[1] = max(0, min(pinky[1], frame_height))


def check_erase():
    erase_bound = 15
    if (abs(thumb[0]-middle[0]) < erase_bound) and (abs(thumb[0]-ring[0]) < erase_bound) and (abs(thumb[1]-middle[1]) < erase_bound) and (abs(thumb[1]-ring[1]) < erase_bound):
        return True
    return False

def check_draw_condition():
    draw_bound = 15
    # If x coord value difference is less than 10 pixels, then they are touched
    if (abs(index[0]-thumb[0]) < draw_bound) and (abs(index[1]-thumb[1]) < draw_bound):
        return True
    return False

def check_draw_condition():
    draw_bound = 15
    # If x coord value difference is less than 10 pixels, then they are touched
    if (abs(index[0]-thumb[0]) < draw_bound) and (abs(index[1]-thumb[1]) < draw_bound):
        return True
    return False

def check_rectangle_pre():
    if (index[0] > rect_l_t[0] and index[0] < rect_r_b[0] and index[1] > rect_l_t[1] and index[1] < rect_r_b[1]):
        return True
    return False

def check_rectangle_post():
    draw_bound = 20
    # If x coord value difference is less than 10 pixels, then they are touched
    if (abs(pinky[0]-thumb[0]) < draw_bound) and (abs(pinky[1]-thumb[1]) < draw_bound):
        return True
    return False

def clear_points():
    erase_pixel_bound = 30
    pop_index = []
    for i in range(len(drawing_list)):
        point = drawing_list[i][0]
        if ((abs(point[0]-index[0])<erase_pixel_bound) and (abs(point[1]-index[1])<erase_pixel_bound)) or \
            ((abs(point[0]-middle[0])<erase_pixel_bound) and (abs(point[1]-middle[1])<erase_pixel_bound)) or \
            ((abs(point[0]-ring[0])<erase_pixel_bound) and (abs(point[1]-ring[1])<erase_pixel_bound)):
            pop_index.append(i)

    for i in sorted(pop_index, reverse=True):
        del drawing_list[i]

def distance(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def display_line(image):
    image = cv2.rectangle(image, rect_l_t, rect_r_b, (0,0,0), 2)

    dist_threshold = 20
    if len(drawing_list) > 1:
        if distance(drawing_list[-2][0], drawing_list[-1][0]) < dist_threshold:
            points_on_line = np.linspace(drawing_list[-2][0], drawing_list[-1][0],
                                         int(np.linalg.norm(np.array(drawing_list[-2][0]) - np.array(drawing_list[-1][0]))),
                                         dtype=np.int)
            for point in points_on_line:
                drawing_list.append([point, selected_color])
        for pixel_cords in drawing_list:
            cv2.circle(image, (pixel_cords[0][0], pixel_cords[0][1]), selected_radius, pixel_cords[1], marker_thickness)

    cv2.imshow("a", image)

# Both the hands should be visible and should be separated by considerable distance
def paint():
    global bkg_img
    global num_frames
    global drawing_list
    global index
    global middle
    global ring
    global thumb
    global pinky
    global selected_color
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.3) as hands:
        while (True):
            ret, image = vid.read()
            # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output and process it with MediaPipe Hands.
            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            image_height, image_width, _ = image.shape

            if check_validity(results)==False:
                image = cv2.flip(image, 1)
                display_line(image)
                # for pixel_cords in drawing_list:
                #     cv2.circle(image, (pixel_cords[0], pixel_cords[1]), selected_radius, selected_color, marker_thickness)
                # if drawing_list:
                #     cv2.polylines(image, [np.array(drawing_list).reshape((-1, 1, 2))], False, selected_color, 2)

            else:
                # frame width is image height and vice versa because of image flip
                annotated_image = cv2.flip(image.copy(), 1)
                draw_flag = False
                erase_flag = False
                color_picker_flag = False
                draw_rectangle_flag = False
                marker_index = 0

                for i in range(len(results.multi_hand_landmarks)):
                    hand_landmarks = results.multi_hand_landmarks[i]
                    calculate_points(hand_landmarks)

                    if check_color_picker():
                        color_picker_flag = True
                        marker_index = abs(i - 1)
                    elif check_rectangle_pre():
                        draw_rectangle_flag = True
                        marker_index = abs(i - 1)
                    elif check_draw_condition():
                        draw_flag = True
                        marker_index = abs(i - 1)
                    elif check_erase():
                        erase_flag = True
                        marker_index = abs(i - 1)

                hand_landmarks = results.multi_hand_landmarks[marker_index]
                calculate_points(hand_landmarks)
                if draw_flag:
                    drawing_list.append([[index[0], index[1]], selected_color])

                elif erase_flag:
                    clear_points()

                elif draw_rectangle_flag:
                    start = []
                    end = []
                    while True:
                        print("rect")
                        ret, image = vid.read()
                        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
                        if (results.multi_hand_landmarks) and len(results.multi_hand_landmarks)>marker_index:
                            hand_landmarks = results.multi_hand_landmarks[marker_index]
                            calculate_points(hand_landmarks)
                            rect_bound = 20
                            if ((abs(pinky[0]-thumb[0]) < rect_bound) and (abs(pinky[1]-thumb[1])) < rect_bound) or len(start)==0 or len(end)==0:
                                print("inside")
                                if len(start) == 0:
                                    start = [index, selected_color]
                                    end = [index, selected_color]
                                    image = cv2.rectangle(image, tuple(start[0]), tuple(end[0]), selected_color, 2)
                                else:
                                    end = [index, selected_color]
                                    image = cv2.rectangle(image, tuple(start[0]), tuple(end[0]), selected_color, 2)
                            else:
                                print("break")
                                # image = cv2.rectangle(image, tuple(start[0]), tuple(end[0]), selected_color, 2)
                                # cv2.imshow("a", cv2.flip(image, 1))
                                break
                        cv2.imshow("a", cv2.flip(image, 1))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                elif color_picker_flag:
                    print("SET")
                    while True:
                        ret, image = vid.read()
                        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
                        if (results.multi_hand_landmarks) and len(results.multi_hand_landmarks)>marker_index:
                            hand_landmarks = results.multi_hand_landmarks[marker_index]
                            calculate_points(hand_landmarks)
                            color_picker_bound = 20
                            if (abs(thumb[0] - middle[0]) < color_picker_bound) and (abs(thumb[1] - middle[1]) < color_picker_bound):
                                selected_color = list(image[thumb[0], thumb[1]])
                                selected_color = (int (selected_color [ 0 ]), int (selected_color [ 1 ]), int (selected_color [ 2 ]))
                                break
                        cv2.imshow("a", cv2.flip(image, 1))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break




                # for pixel_cords in drawing_list:
                #     cv2.circle(annotated_image, (pixel_cords[0], pixel_cords[1]), selected_radius, selected_color, marker_thickness)

                # if drawing_list:
                #     cv2.polylines(annotated_image, [np.array(drawing_list).reshape((-1, 1, 2))], False, selected_color, 2)

                display_line(annotated_image)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

paint()
