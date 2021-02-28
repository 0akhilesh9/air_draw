import cv2
import os
import numpy as np
import mediapipe as mp

import gesture as gesture

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
thickness = 2
selected_radius = 3
drawing_list = []
selected_color = [0, 0, 255]
num_frames = 0
buffer_pixels = 20
bkg_img = None



def paint():
    global bkg_img
    global num_frames
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4) as hands:
        while (True):
            ret, image = vid.read()
            # Convert the BGR image to RGB, flip the image around y-axis for correct
            # handedness output and process it with MediaPipe Hands.
            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            image_height, image_width, _ = image.shape
            # Print handedness (left v.s. right hand).
            # print(results.multi_handedness)

            # Draw hand landmarks of each hand.
            if not results.multi_hand_landmarks:
                cv2.imshow("a", cv2.flip(image, 1))
            else:
                # frame width is image height and vice versa because of image flip
                annotated_image = cv2.flip(image.copy(), 1)
                for hand_landmarks in results.multi_hand_landmarks:
                    index_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                    index_x = max(0, min(index_x, frame_width))
                    index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                    index_y = max(0, min(index_y, frame_height))
                    drawing_list.append([index_x, index_y])

                    print("{0} {1} {2} {3}".format(index_x, index_y, frame_width, frame_height))

                    rect_left_x = int(min([point.x for point in hand_landmarks.landmark]) * image_width) - buffer_pixels
                    rect_left_x = max(0, min(rect_left_x, frame_width))
                    rect_top_y = int(min([point.y for point in hand_landmarks.landmark]) * image_height) - buffer_pixels
                    rect_top_y = max(0, min(rect_top_y, frame_height))
                    rect_right_x = int(max([point.x for point in hand_landmarks.landmark]) * image_width) + buffer_pixels
                    rect_right_x = max(0, min(rect_right_x, frame_width))
                    rect_bottom_y = int(max([point.y for point in hand_landmarks.landmark]) * image_height) + buffer_pixels
                    rect_bottom_y = max(0, min(rect_bottom_y, frame_height))

                    cropped_img = annotated_image[rect_top_y : (rect_top_y+(rect_bottom_y-rect_top_y)), rect_left_x : (rect_left_x+(rect_right_x-rect_left_x))]
                    thresholded, bkg_img, num_frames = gesture.get_gesture(cropped_img, bkg_img, num_frames)

                    # Print index finger tip coordinates.
                    # print(
                    #     f'Index finger tip coordinate: (',
                    #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                    # )
                    # mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    cv2.rectangle(annotated_image, (rect_left_x, rect_top_y), (rect_right_x, rect_bottom_y), selected_color, thickness)
                    cv2.imshow("a", thresholded)

                for pixel_cords in drawing_list:
                    cv2.circle(annotated_image, (pixel_cords[0], pixel_cords[1]), selected_radius, selected_color,
                               thickness)
                    # annotated_image[pixel_cords[1], pixel_cords[0]] = selected_color  #(x,y)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

paint()