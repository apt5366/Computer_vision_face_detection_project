import sys
import time
import math
import inspect
import threading
import cv2
import numpy as np
import PID
import hiwonder.Board as Board
import hiwonder.Camera as Camera
from hiwonder import yaml_handle

################################################################################
# Configuration variables
################################################################################
# Window properties
esc_key = 27
window_name = "Face detection"
image_size = (640, 480) # Don't change this
image_center = (320, 240) # Don't change this

# Text and shape properties
text_font = cv2.FONT_HERSHEY_DUPLEX
text_pos_line1 = (4, 16)
text_pos_line2 = (4, 36)
text_scale = 0.5
line_color_in = (63, 63, 255)
line_color_out = (0, 0, 0)
line_thick_in = 1
line_thick_out = 3
circle_color_target = (255, 63, 63)
circle_color_center = (63, 255, 63)
circle_color_out = (0, 0, 0)
circle_radius_in = 8
circle_radius_out = 10

# Resolution for updating framerate
framerate_res = 10

# Global data variables
faces = []
img = []
img_copy = []
faces_copy = []

# PID Controller objects
pid_pan = PID.PID(P=0.45, I=0.4, D=0.1) # TODO Create a PID controller object for the pan servo
pid_tilt = PID.PID(P=0.45, I=0.4, D=0.1) # TODO Create a PID controller object for the tilt servo

# The classifier
# classifier = None # TODO Load the Haar cascade classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # TODO Load the Haar cascade classifier

# Servo ID handles
HEAD_PAN = 2
HEAD_TILT = 1

# Servo default position
default_pan = 1500
default_tilt = 1500

# Servo current positions
cur_pan = default_pan
cur_tilt = default_tilt

# Servo bounds
bounds_pan = (600, 2400)
bounds_tilt = (1000, 2000)

# Servo movement values
move_time = 150


################################################################################
# Helper functions
################################################################################
# Set the tilt motor
def set_tilt(value, timing):
    new_val = clamp(value, bounds_tilt)
    Board.setPWMServoPulse(HEAD_TILT, new_val, timing)

# Set the pan motor
def set_pan(value, timing):
    new_val = clamp(value, bounds_pan)
    Board.setPWMServoPulse(HEAD_PAN, new_val, timing)

# Clamp a value to a range
def clamp(value, bounds):
    out_val = bounds[0] if value < bounds[0] else value
    out_val = bounds[1] if out_val > bounds[1] else out_val
    return out_val

# A more accurate sleep function
# Kind of hacky, but it avoids the overhead on time.sleep()
def my_sleep(sleep_time):
    if sleep_time < 0:
        sys.exit("Negative value passed to {}; value must be nonnegative!".format(inspect.stack()[0][3]))
    end_time = time.time() + sleep_time
    while time.time() <= end_time:
        pass

# Get the Euclidean distance between two points
def get_dist(pt_a, pt_b):
    diff_x = pt_a[0] - pt_b[0]
    diff_y = pt_a[1] - pt_b[1]
    return math.sqrt(diff_x * diff_x + diff_y * diff_y)

# Get the center point of a face
def get_face_center(face):
    x= (face[0]+((face[2])/2)) # (top_left_x + (width/2)) 
#     y= ((face[1]+face[3])/2) # (top_left_y+ height) / 2
    y= (face[1]+((face[3])/2)) # (top_left_x + height/2)
    return (int(x), int(y)) # TODO Replace temporary return value

# Get the largest face from a list of faces
def get_largest_face(face_list):
    # TODO Implement this function
    # If no faces in list, return an empty list
    length= len(face_list)
    if(length==0):
        return []

    # Otherwise, find and return the largest face (use width * height as size)
    max_size = 0
    max_size_index=0
    index = -1
    # size of any face is width*height of the face
    for faces in face_list:
        index = index + 1
        if ((faces[2] * faces[3]) > max_size):
            max_size = (faces[2] * faces[3])
            max_size_index= index


    return face_list[max_size_index]

# Get the face closest to center of the screen from a list of faces
def get_centermost_face(face_list):
    # TODO Implement this function
    # If no faces, return an empty list
    length = len(face_list)
    if (length == 0):
        return []

    # Otherwise, find and return the face whose center is closest to the image center

    min_dist_to_centre = 0
    req_index = 0
    index = -1

    min_dist_to_centre = (get_dist(image_center, get_face_center(face_list[0])))
    for faces in face_list:
        index = index + 1
        if ((get_dist(image_center, get_face_center(faces))) < min_dist_to_centre):
            min_dist_to_centre = (get_dist(image_center, get_face_center(faces)))
            req_index = index


    return face_list[req_index]

# Select which method to use for finding the "best" face to track
# Basically an alias so you don't have to change function calls at various places in your code
# get_best_face = get_centermost_face
get_best_face = get_largest_face


################################################################################
# Motion control functions
################################################################################
# Reset the camera and PID controllers
def reset_motion():
    # Reset the motors
    global cur_pan, cur_tilt
    cur_pan = default_pan
    cur_tilt = default_tilt
    set_pan(cur_pan, 750)
    set_tilt(cur_tilt, 750)
    my_sleep(0.8)

    # Reset the PID controllers
    # TODO reset the PID controllers here
    pid_pan.clear()
    pid_tilt.clear()


# The function that actually does the motion based on PID control
def do_motion(faces):
    # Make sure to use global variables
    global cur_pan, cur_tilt

    # Get the best face using chosen criteria
    best_face = get_best_face(faces)
    
    # If there is no face to track, simply return
    if len(best_face) == 0:
        return
    
    # TODO Get the center of the best face
    best_face_center= get_face_center(best_face)

    # TODO Update the pan PID controller
    pid_pan.SetPoint = image_center[0]
    pid_pan.update(best_face_center[0])
    delta_pan= int(pid_pan.output)
    cur_pan= clamp(cur_pan+delta_pan, bounds_pan)

    # TODO Update the tilt PID controller
    pid_tilt.SetPoint = image_center[1]
    pid_tilt.update(best_face_center[1])
    delta_tilt= int(pid_tilt.output)
    cur_tilt= clamp(cur_tilt+delta_tilt, bounds_tilt)

    # TODO Set the motor positions
    set_pan(cur_pan,move_time)
    set_tilt(cur_tilt,move_time)
    


################################################################################
# Face detection function (run as separate thread)
################################################################################
def face_detection(stop_sig, img_lock, face_cond):
    # Use global variable for faces list
    global faces

    # Keep looping until recieving a signal to stop
    while not stop_sig.is_set():
        # Try to do the face detection
        try:
            # Require grayscale image, need access to image
            with img_lock:
                g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Scale the image down to half the size for ease of computation
            g_img = cv2.resize(g_img, (320, 240), interpolation=cv2.INTER_NEAREST)
            g_img = cv2.GaussianBlur(g_img, (3, 3), 3)

            # Require that this thread is the only one accessing faces list
            with face_cond:
                # Do face detection
                # temp_faces = [] # TODO Do multiscale detection to get a list of faces here, replace placeholder
                temp_faces = classifier.detectMultiScale(g_img, 1.1, 6) # TODO Do multiscale detection to get a list of faces here, replace placeholder

                # Because we scaled down for classification, we need to scale our results
                faces = [[x * 2 for x in y] for y in temp_faces]
                # Do the PID controller updates
                do_motion(faces)
                # Block thread until woken up - makes sure main thread can copy data before writing new data back
                face_cond.wait()

        # If an exception is thrown, skip this loop iteration and try again
        except:
            pass


################################################################################
# Main code, sets up worker threads and runs main thread
################################################################################
if __name__ == '__main__':
    # Set up camera
    open_once = yaml_handle.get_yaml_data('/boot/camera_setting.yaml')['open_once']
    is_cv2cam = False
    if open_once:
        camera = cv2.VideoCapture('http://127.0.0.1:8080/?action=stream?dummy=param.mjpg')
        is_cv2cam = True
    else:
        camera = Camera.Camera(image_size)
        camera.camera_open()

    # Create the window
    cv2.namedWindow(window_name)

    # Create event for notifying worker threads to stop
    stop_sig = threading.Event()

    # Create mutual exclusion locks
    img_lock = threading.Lock()
    face_cond = threading.Condition()

    # Create thread for doing face detection
    face_thread = threading.Thread(target=face_detection, args=(stop_sig, img_lock, face_cond, ))

    # Variables for handling frames
    framerate = 0.0
    fps_counter = 0
    start = time.time() # First frame time

    # Reset the head servos and PID controllers
    reset_motion()
    
    # Start the face detection thread
    face_thread.start()

    # The main loop
    while True:
        # Get the frame, need image mutex lock
        with img_lock:
            read_success, img = camera.read()
            if read_success and img is not None:
                img_copy = img.copy()

        # Frame is good
        if read_success and img_copy is not None:
            # Clone faces if mutex lock is available
            if face_cond.acquire(blocking = False):
                # If there's at least one face, copy the list, otherwise reset to empty list
                if len(faces) > 0:
                    faces_copy = faces.copy()
                else:
                    faces_copy = []
                # Notify waiting threads and then release the lock
                face_cond.notifyAll()
                face_cond.release()

            # Put a circle in the middle of the screen
            cv2.circle(img_copy, image_center, circle_radius_out, circle_color_out, -1, cv2.LINE_AA)
            cv2.circle(img_copy, image_center, circle_radius_in, circle_color_center, -1, cv2.LINE_AA)

            # Render boxes around most recent set of faces
            for face in faces_copy:
                cv2.rectangle(img_copy, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), line_color_out, line_thick_out)
                cv2.rectangle(img_copy, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), line_color_in, line_thick_in)

            # Render a circle on the face we want to track
            best_face = get_best_face(faces_copy)
            if len(best_face) > 0:
                cv2.circle(img_copy, get_face_center(best_face), circle_radius_out, circle_color_out, -1, cv2.LINE_AA)
                cv2.circle(img_copy, get_face_center(best_face), circle_radius_in, circle_color_target, -1, cv2.LINE_AA)

            # Add framerate to output image
            cv2.putText(img_copy, '{:.2f} FPS - Press ESC to quit'.format(framerate), text_pos_line1, text_font, text_scale, line_color_out, line_thick_out, cv2.LINE_AA)
            cv2.putText(img_copy, '{:.2f} FPS - Press ESC to quit'.format(framerate), text_pos_line1, text_font, text_scale, line_color_in, line_thick_in, cv2.LINE_AA)

            # Add number of faces detected to image
            cv2.putText(img_copy, 'Number of faces detected: {}'.format(len(faces_copy)), text_pos_line2, text_font, text_scale, line_color_out, line_thick_out, cv2.LINE_AA)
            cv2.putText(img_copy, 'Number of faces detected: {}'.format(len(faces_copy)), text_pos_line2, text_font, text_scale, line_color_in, line_thick_in, cv2.LINE_AA)

            # Render image to screen
            cv2.imshow(window_name, img_copy)

            # Compute FPS every so often
            fps_counter += 1
            if fps_counter >= framerate_res:
                end = time.time()
                framerate = (framerate_res / (end - start))
                fps_counter = 0
                start = end
        
        # Error: getting frame failed, skip frame        
        else:
            pass

        # Check to see if a key was pressed this frame
        k = cv2.waitKey(1)
        # ESC pressed, exit loop
        if k % 256 == esc_key:
            print("ESC pressed, closing...")
            break

    # Signal to other threads that program is ending
    stop_sig.set()

    # Notify face detection thread to stop waiting
    with face_cond:
        face_cond.notifyAll()
    face_thread.join()

    # Cleanup
    if is_cv2cam:
        camera.release()
    else:
        camera.camera_close()
    cv2.destroyAllWindows()