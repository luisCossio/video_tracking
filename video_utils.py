import os
import cv2
import numpy as np

from datetime import timedelta


def format_timedelta(td: timedelta):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap: cv2.VideoCapture, saving_fps: int):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s


def rotate_image(img,positive=False):
    if positive:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

class frame_extractor:
    def __init__(self, video_file: str, extraction_fps=1, size=[1024, 1024], rotate=False, split = [1,1]):
        """
        Class to handle video frame extraction. It provide a series of frames, with a relative

        :param video_file: path to video file (.MOV or .AVI)
        :param extraction_fps: frame rate of provided detection
        :param size:
        :param split: partitions for height and width dimension
        """
        self.size = size
        self.extraction_fps = extraction_fps
        self.count = 0
        self.video_file = video_file
        self.cap = None
        self.video_ended = False
        self.rotate = rotate
        self.split = split

    def start(self, video_file=''):
        """
        Module to start the sequence frame extraction. It can be provided with a new video file to extract from.
        Restart previous video sequence otherwise.

        :param video_file:
        :return:
        """
        # if the self.saving_fps is above video FPS, then set it to FPS (as maximum)

        self.count = 0
        self.video_ended = False
        if video_file:
            self.video_file = video_file
        self.cap = cv2.VideoCapture(self.video_file)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        saving_frames_per_second = min(self.fps, self.extraction_fps)
        # get the list of duration spots to save
        self.saving_frames_durations = get_saving_frames_durations(self.cap, saving_frames_per_second)

    def image_tranformation(self, frame: np.ndarray):
        """
        Module to standardize frame data. Returns modified image.
        :param frame: image data
        :return:
        """
        # frame = self.cut_image(frame)
        if self.rotate:
            frame = rotate_image(frame)
        # frame = self.scaling(frame)

        return frame


    def cut_image(self, frame: np.ndarray):
        """
        Cut image in order to fit (if possible) desired dimensions. Returned cropped image.
        :param frame: image data.
        :return:
        """
        shape_img = frame.shape
        center = shape_img[0] // 2, shape_img[1] // 2
        coordinates = np.array([[center[0] - self.size[0] // 2, center[0] + self.size[0] // 2],
                                [center[1] - self.size[1] // 2, center[1] + self.size[1] // 2]])
        ylim = np.clip(coordinates[0, :], 0, shape_img[0])
        xlim = np.clip(coordinates[1, :], 0, shape_img[1])

        cut_img = frame[ylim[0]:ylim[1], xlim[0]:xlim[1], :]
        return cut_img

    def get_next(self):
        """
        Method to get next frame in the video sequence, according to the extraction frame rate.
        :return: list of images, split accordingly
        """
        if self.video_ended:
            return None

        is_read, frame = self.cap.read()

        if not is_read:  # no more frames
            self.video_ended = True
            return None
        else:  # get frame
            # get the duration by dividing the frame count by the FPS
            frame_duration = self.count / self.fps
            try:
                # get the earliest duration to save
                closest_duration = self.saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                print("No more frames available.")
                return None
            if frame_duration >= closest_duration:
                # if closest duration is less than or equals the frame duration,
                # then save the frame
                # frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
                # cv2.imwrite(os.path.join(filename, f"frame{frame_duration_formatted}.JPG"), frame)
                # drop the duration spot from the list, since this duration spot is already saved
                try:
                    self.saving_frames_durations.pop(0)
                except IndexError:
                    print("ERROR: Invalid frame read attempt")
                    pass
            # increment the frame count
            self.count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.image_tranformation(frame)


def get_cuts(image, shape, splits):
    cuts = []
    cut_locations = []
    image_shape = np.array(image.shape[:2], dtype=np.int16)
    splits_arr = np.array(splits)
    splits_arr[splits_arr<2] = 2
    distances = ((image_shape - np.array(shape)) / (splits_arr - 1)).astype(np.int16)

    for i in range(splits[0]):
        start_y = distances[0] * i
        cuts += [[]]
        cut_locations += [[]]
        for j in range(splits[1]):
            cut = [start_y, distances[1] * j]
            cut_img = cut_image(image, shape, cut)
            cuts[-1] += [cut_img]
            cut_locations[-1] += [cut]

    return cuts, cut_locations

def cut_image(img: np.ndarray, shape_cut: list, start: list):
    """
    Function to get a cut from an image, based on the shape and starting location.
    :param img: image array
    :param shape_cut: List or array of length 2 for y and x dimensions of cut
    :param start: List or array of length 2 for y and x starting position
    :return:
    """

    end = [start[0] + shape_cut[0], start[1] + shape_cut[1]]
    return img[start[0]:end[0], start[1]:end[1]]



def draw_image(image:np.ndarray, predictions:np.ndarray, colors:list, number_classes=1, thickness=2):
    """
    Function to draw a rectangle in a given image
    :param image: numpy array / image
    :param predictions: matrix of predictions array as (x1,y1,x2,y2,confidence,class)
    :param colors: list of colors to paint each class
    :param number_classes: number of classes found in class column
    :param thickness: thickness of rectangles.
    :return:
    """
    if len(predictions) == 0:
        return image
    predictions = predictions.reshape([-1, 6])
    valid_cat = [i for i in range(number_classes)]
    for predicted_object in predictions:
        class_predicted = int(predicted_object[5])
        if class_predicted in valid_cat:
            start_point = (round(predicted_object[0]), round(predicted_object[1]))
            end_point = (round((predicted_object[2])), round((predicted_object[3])))
            image = cv2.rectangle(image, start_point, end_point, colors[class_predicted], thickness)
    return image

def draw_image_from_predictions(image:np.ndarray, predictions:np.ndarray, classes_predicted,colors:list,
                                number_classes=1, thickness=2):
    """
    Function to draw a rectangle in a given image
    :param image: numpy array / image
    :param predictions: matrix of predictions array as (x1,y1,x2,y2)
    :param colors: list of colors to paint each class
    :param number_classes: number of classes found in class column
    :param thickness: thickness of rectangles.
    :return:
    """
    if len(predictions) == 0:
        return image
    predictions = predictions.reshape([-1, 4])
    valid_cat = [i for i in range(number_classes)]
    for i,predicted_object in enumerate(predictions):
        class_predicted = int(classes_predicted[i])
        if class_predicted in valid_cat:
            start_point = (round(predicted_object[0]), round(predicted_object[1]))
            end_point = (round((predicted_object[2])), round((predicted_object[3])))
            image = cv2.rectangle(image, start_point, end_point, colors[class_predicted], thickness)
    return image

