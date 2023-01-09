import os
import cv2
import yaml
import argparse
import numpy as np

from dlib import matrix, max_cost_assignment

# from models_resources.models_yolo.yolo import Model
# from utils_models.utils_yolo.torch_utils import intersect_dicts
# import video_utils as vu
import video_detector as vd
import video_utils as vu


# from abc import ABCMeta, abstractmethod


class Kalman_filter:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, cov_model_diag=1.0,
                 cov_obs_diag=1.0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = np.eye(len(A)) * np.array([cov_model_diag])
        self.R = np.eye(len(C)) * np.array([cov_obs_diag])

    def prior(self, x, u):
        return self.A @ x + self.B @ u

    def observation(self, x, u):
        return self.C @ x + self.D @ u

    def posterior(self, x, prior_x, u, observation, P):
        z_prior = self.observation(x, u)  # estimation
        prior_P = self.A @ P @ self.A + self.Q
        W = prior_P @ self.C.transpose() @ np.linalg.inv(self.C @ prior_P @ self.C.transpose() + self.R)  # Kalman gain
        post_x = prior_x + W @ (observation - z_prior)  # posterior
        post_P = (np.eye(len(P)) - W @ self.C) @ prior_P  #
        return post_x, post_P


def calculate_manhattan_distance(matrix1, matrix2):
    """
    Calculate absolute distance between locations
    :param matrix1:
    :param matrix2:
    :return:
    """
    distance = np.abs(matrix1[:, None, :] - matrix2)
    distance = np.sum(distance, axis=2)
    return distance


def calculate_iou_distance(matrix1, matrix2):
    """
    Calculate IoU correlation between locations in matrices
    :param matrix1: Position matrix array as [x1,y1,x2,y2] columns
    :param matrix2: Position matrix array as [x1,y1,x2,y2] columns
    :return:
    """
    inter = np.maximum(np.minimum(matrix1[:, None, 2], matrix2[:, 2]) - np.maximum(matrix1[:, None, 0], matrix2[:, 0]),
                       0)
    inter *= np.maximum(np.minimum(matrix1[:, None, 3], matrix2[:, 3]) - np.maximum(matrix1[:, None, 1], matrix2[:, 1]),
                        0)
    eps = 0.000001

    # Union Area
    w1, h1 = matrix1[:, 2] - matrix1[:, 0], matrix1[:, 3] - matrix1[:, 1] + eps
    w2, h2 = matrix2[:, 2] - matrix2[:, 0], matrix2[:, 3] - matrix2[:, 1] + eps
    union = ((w1 * h1)[:, None] + w2 * h2) - inter + eps

    iou = inter / union
    return iou


def format_detections(detections: np.ndarray, speed=0):
    """
    Function to format detections into a position format
    :param detections: detection array to be formatted into a matrix of columns [x1,y1,x2,y2,x'1,y'1,x'2,y'2]. Initial
    speed is zero.
    :return:
    """
    detection = np.zeros([len(detections), 8])
    detection[:, :4] = detections[:, :4]
    detection[:, 4:] += speed
    return detection


def save_detections(file_name: str, new_data: np.ndarray):
    if os.path.isfile(file_name):
        file = np.loadtxt(file_name)
        zero_row = np.zeros([1, 6])
        if type(new_data) == list:
            print("Erro found: ")
        formatted_data = np.concatenate([file, new_data.reshape([-1, 6]), zero_row])
        np.savetxt(file_name, formatted_data)
    else:
        zero_row = np.zeros([1, 6])
        formatted_data = np.concatenate([new_data, zero_row])
        np.savetxt(file_name, formatted_data)


class video_tracker:
    def __init__(self, analizer: vd.analizer, configs: dict, absent_frame=9):
        """
        Class
        :param analizer:
        :param configs:
        :param absent_frame:
        """
        self.video_analizer = analizer
        self.records_seen = None  # records of objects already seen and out of view
        self.records_visible = None  # records of objects currently on view
        self.covariance_matrices = []
        self.absent_frame = absent_frame

        self.alpha_speed = configs['speed_weight']
        self.threshold_iou = configs['thr_iou']
        self.n_classes = configs['n_classes']
        self.min_valid = configs['min_detections']
        assert (len(configs['initial_speed']) == 4 or len(configs['initial_speed']) == 2), "Invalid length"
        if len(configs['initial_speed']) == 2:
            configs['initial_speed'] += configs['initial_speed']
        self.initial_speed = np.array(
            configs['initial_speed'])  # list of speed for all items arange as [x'1,y'1,x'2,y'2]
        self.configs = configs

        self.kalman_filter = self.get_kalman_filter()

    def save_tracking_records(self, path_video: str, save_file: str):
        self.video_analizer.start_video_predictions(path_video)
        self.start_records(self.video_analizer.get_next_prediction())
        while self.video_analizer.is_not_over():
            predictions = self.video_analizer.get_next_prediction()
            self.update_records(predictions)
        self.save_file(save_file)

    def save_tracking_video_records(self, path_video, save_file):
        self.video_analizer.start_video_predictions(path_video)
        frame, detection = self.video_analizer.get_next_frame_prediction()
        self.start_records(detection)
        frames_tracked = []
        # frames = []
        while self.video_analizer.is_not_over():
            frame, predictions = self.video_analizer.get_next_frame_prediction()
            self.update_records(predictions)
            frames_tracked += [self.draw_current_objects(frame)]

        self.save_file(save_file)
        self.save_video_file(frames_tracked, save_file.replace('.csv', '.avi'))

    def save_video_file(self, frames: list[np.ndarray], save_file='records.csv'):
        """
        Method to save the records of total fruits seen.

        :param save_file:  path to save file
        :return:
        """
        width = frames[0].shape[1]
        height = frames[0].shape[0]
        size = (width, height)
        codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        out = cv2.VideoWriter(save_file, codec, 10, size)
        # out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

        print("Building Video:\n {:s}".format(save_file))
        for i in range(len(frames)):
            out.write(frames[i])
        out.release()

    def save_file(self, save_file='records.avi'):
        """
        Method to save the records of total fruits seen.

        :param save_file:  path to save file
        :return:
        """
        records = np.concatenate(self.final_records, axis=0)
        np.savetxt(save_file, records)

    def draw_current_objects(self, frame):
        valid_bbs = self.records_visible[:, 2] > 1
        class_bb = self.records_visible[valid_bbs, 4:]
        class_bb[:, 1] *= 2
        class_bb = np.argmax(class_bb, axis=1)
        boxes = self.position_visible[valid_bbs, :4]
        img_save = vu.draw_image_from_predictions(frame.copy(), boxes, classes_predicted=class_bb,
                                                  colors=self.configs['color_classes'],
                                                  number_classes=self.n_classes)
        return cv2.cvtColor(img_save, cv2.COLOR_BGR2RGB)

    def start_records(self, initial_prediction: np.ndarray):
        """
        Method to initialize predictions using detections

        :param initial_prediction: matrix of shape [N,6] for the columns x1,y1,x2,y2, conf, class
        :return:
        """
        # save_detections(file_name='records_detection.cvs', new_data=initial_prediction)  # DELETE
        n_objects = len(initial_prediction)
        # if n_objects > 0:
        self.final_records = []  # categorical features

        self.records_visible = np.zeros([n_objects, 4 + self.n_classes],
                                        dtype=np.int32)  # categorical features

        self.records_visible[:, 0] = 0  # first seen
        self.records_visible[:, 1] = 0  # final seen
        self.records_visible[:, 2] = 1  # object visible, for debbuging and video drawing
        self.records_visible[:, 3] = 0  # valid objects. All invalid since not enough frames confirm their validity

        rows = np.arange(n_objects)
        classes_detected = initial_prediction[:, 5].astype(np.int32)
        self.records_visible[rows, 4 + classes_detected] = 1  # class
        self.mean_speed = np.zeros([n_objects, 4], dtype=np.float32) + self.initial_speed

        # self.position_objects = np.empty([len(initial_prediction), 8])
        self.position_visible = format_detections(initial_prediction[:, :4], speed=self.initial_speed)
        self.covariance_matrices = np.zeros([n_objects, 8, 8]) + np.eye(8) * self.configs[
            'variance_model']  # P confusion matrix
        self.frame_count = 1
        self.average_speeds = [np.mean(self.mean_speed)]

    def update_records(self, new_detections: np.ndarray):
        """
        Method to update records

        :param new_detections: current detections
        :return:
        """
        # save_detections(file_name='records_detection.cvs',new_data=new_detections) # DELETE
        prior_locations = self.get_update_kalman(self.position_visible, self.mean_speed)  # get kalman prediction
        if len(new_detections) == 0:
            self.no_detection_protocol(prior_locations)

        else:
            new_detections_positions = format_detections(new_detections[:, :4], np.mean(self.mean_speed, axis=0))
            indices_records, indices_detections = self.connect_locations(prior_locations[:, :4],
                                                                         new_detections_positions[:, :4],
                                                                         self.threshold_iou)  # compare

            # update current detection
            indices_records_not_found = np.delete(np.arange(len(self.records_visible)), indices_records)

            self.update_positions(prior_locations, new_detections_positions, indices_records, indices_records_not_found,
                                  indices_detections)

            no_matches_detections = np.delete(np.arange(len(new_detections)), indices_detections)

            self.update_states(new_detections[:, 4:], indices_records, indices_detections, indices_records_not_found)

            self.add_new_objects(new_detections_positions[no_matches_detections],
                                 state_detections=new_detections[no_matches_detections, 4:])

        self.frame_count += 1

    def no_detection_protocol(self, prior_locations):
        indices_records = np.arange(len(prior_locations))
        self.update_prior_locations(prior_locations, indices_records)
        indices_records_not_found = np.empty([0], dtype=np.int32)
        # indices_detections = np.empty([0], dtype=np.int32)
        # self.update_states(new_detections, indices_records, indices_detections, indices_records_not_found)
        self.clean_unseen_objects(indices_records_not_found)
        if len(self.mean_speed) > 0: # DELETE
            self.average_speeds += [np.mean(self.mean_speed)]
        else:
            self.average_speeds += [0]

    def update_positions(self, prior_locations: np.ndarray, new_detections: np.ndarray, indices_prior: np.ndarray,
                         indices_not_seen: np.ndarray, indices_detections: np.ndarray):

        old_positions = self.position_visible.copy()
        # speed observation based on diference between observation and previous location
        new_detections[indices_detections, 4:] = new_detections[indices_detections, :4] - old_positions[indices_prior,
                                                                                          :4]
        for i, idx_detect in enumerate(indices_detections):
            resulting_positions, updated_covariance = self.kalman_filter.posterior(
                x=self.position_visible[indices_prior[i]], prior_x=prior_locations[indices_prior[i]],
                u=self.mean_speed[indices_prior[i]],
                observation=new_detections[idx_detect], P=self.covariance_matrices[indices_prior[i]])
            self.covariance_matrices[indices_prior[i]] = updated_covariance
            self.position_visible[indices_prior[i]] = resulting_positions

        self.update_prior_locations(prior_locations, indices_not_seen)
        self.update_speed(old_positions)

    def update_prior_locations(self, prior_locations: np.ndarray, indices_not_seen: np.ndarray):
        self.position_visible[indices_not_seen] = prior_locations[indices_not_seen]
        self.covariance_matrices[indices_not_seen] += np.eye(8) * self.configs['variance_model']

    def update_speed(self, old_positions: np.ndarray):
        new_position = self.position_visible[:, :4].copy()
        new_speed = new_position[:, :4] - old_positions[:, :4]
        self.mean_speed = self.alpha_speed * new_speed + (1 - self.alpha_speed) * self.mean_speed
        self.average_speeds += [np.mean(self.mean_speed)]

    def update_states(self, detections_data: np.ndarray, indices_records: np.ndarray, indices_detections: np.ndarray,
                      indices_records_not_found: np.ndarray):
        """
        Method to update previously seen objects. Classes and visibility fields are updated
        :param detections_data: detection of this frame
        :param indices_records: indices of matching records
        :param indices_records_not_found: indices of records not seen in this frame
        :return:
        """
        classes_data = detections_data[:, 1].astype(np.int32) + 4
        self.records_visible[indices_records, classes_data[indices_detections]] += 1
        self.records_visible[indices_records, 1] = self.frame_count  # final seen
        self.records_visible[indices_records, 2] += 1  # object visible, for debbuging and video drawing
        valid_indices = indices_records[self.records_visible[indices_records, 2] > self.min_valid]
        self.records_visible[valid_indices, 3] = 1  # valid items
        self.clean_unseen_objects(indices_records_not_found)

    def add_new_objects(self, position_predictions: np.ndarray, state_detections: np.ndarray):
        """
        Method to add newly seen items to be new objects
        :param position_predictions: predictions positions to be added.
        :param state_detections: class and confidence matrix [conf, class]
        :return:
        """

        position_predictions = position_predictions.reshape([-1, 8])
        state_detections = state_detections.reshape([-1, 2])
        n_detections = len(state_detections)
        new_state_records = np.zeros([n_detections, 4 + self.n_classes], dtype=np.int32)

        new_state_records[:, 0] = self.frame_count  # first seen
        new_state_records[:, 1] = self.frame_count  # final seen
        new_state_records[:, 2] = 1  # object visible, for debbuging and video drawing
        # new_state_records[:, 3] = 0  # valid objects. All invalid since not enough frames confirm their validity

        rows = np.arange(len(new_state_records))
        classes_detected = state_detections[:, 1].astype(np.int32)
        new_state_records[rows, 4 + classes_detected] = 1  # confidence and class

        self.records_visible = np.concatenate([self.records_visible, new_state_records])
        self.position_visible = np.concatenate([self.position_visible, position_predictions])

        new_speed = np.empty([n_detections, 4])
        new_speed[:] = np.mean(self.mean_speed, axis=0)
        new_covs = np.empty([n_detections, 8, 8])
        new_covs[:] = np.eye(8) * self.configs['variance_model']
        self.mean_speed = np.concatenate([self.mean_speed, new_speed])
        self.covariance_matrices = np.concatenate([self.covariance_matrices, new_covs])

    def get_update_kalman(self, current_locations: np.ndarray, mean_speed: np.ndarray):
        input_speed = current_locations[:, 4:] - mean_speed
        predictions = self.kalman_filter.prior(x=current_locations.transpose(), u=input_speed.transpose()).transpose()
        return predictions

    def get_kalman_filter(self):
        A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])

        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        C = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])

        D = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]])

        return Kalman_filter(A, B, C, D, self.configs['variance_model'], self.configs['variance_observation'])

    def connect_locations(self, positions1: np.ndarray, positions2: np.ndarray, threshold_iou: float):
        min_value = 0.1
        scores = calculate_iou_distance(positions1, positions2)
        rows1 = len(positions1)
        rows2 = len(positions2)
        padded_size = max(rows1, rows2)
        padded_scores = np.zeros([padded_size, padded_size], dtype=np.float32)
        matches = min(rows2, rows1)
        if rows1 > rows2:
            padded_scores[:, :rows2] = scores
            padded_scores[:, rows2:] = min_value

        else:
            padded_scores[:rows1, :] = scores
            padded_scores[rows1:, :] = min_value

        scores = matrix(padded_scores)
        assignment = max_cost_assignment(scores)
        rows = np.arange(rows1)
        if rows1 > rows2:
            assignment = np.array(assignment)
            valid_ids = assignment < matches
            matches_ids2 = assignment[valid_ids]
            matches_ids1 = rows[valid_ids]
            valid_ids = padded_scores[matches_ids1, matches_ids2] > threshold_iou
            matches_ids1 = matches_ids1[valid_ids]
            matches_ids2 = matches_ids2[valid_ids]
        else:
            valid_ids = padded_scores[rows, assignment[:matches]] > threshold_iou
            matches_ids1 = rows[valid_ids]
            matches_ids2 = np.array(assignment[:matches])[valid_ids]
        return matches_ids1, matches_ids2

    def clean_unseen_objects(self, indices_unseen: np.ndarray):
        """
        Method to delete extract rows from the records. Either because are valid targets, no longer visible visible
        or because are invalid detections. Records are arrange as matrices with columns [N frame first appearance,
        N frame last appearance, N times seen, valid status, Class1, Class2, ...].

        :param indices_unseen: indices of objects not seen in the last frame, but still being accounted for.
        :return:
        """
        self.records_visible[indices_unseen, 2] -= 2
        indices_save = (self.records_visible[indices_unseen, 2] < -2) & (self.records_visible[indices_unseen, 3] == 1)
        indices_ignore = np.delete(np.arange(len(self.records_visible)), indices_unseen[indices_save])
        self.save_valid_objects(indices_unseen[indices_save], indices_ignore)
        self.delete_invalid_objects()

    def save_valid_objects(self, indices_save: np.ndarray, indices_not_save: np.ndarray):
        """
        Save records of object no longer visible.

        :param indices_save: indices of items to be elected
        :return: None
        """
        if len(indices_save) > 0:
            records = self.records_visible[indices_save]
            records[:, 5] *= 2  # class unripe is consider a more likely scenario
            records[:, 4] = np.argmax(records[:, 4:], axis=1)

            self.final_records += [records[:, :5].copy()]

            self.records_visible = self.records_visible[indices_not_save]  # delete saved rows.
            self.position_visible = self.position_visible[indices_not_save]
            self.covariance_matrices = self.covariance_matrices[indices_not_save]
            self.mean_speed = self.mean_speed[indices_not_save]

    def delete_invalid_objects(self):
        valid_items = (self.records_visible[:, 2] > -5) & (self.position_visible[:, 0] > - 10.0)
        self.records_visible = self.records_visible[valid_items]  # delete saved rows.
        self.position_visible = self.position_visible[valid_items]
        self.covariance_matrices = self.covariance_matrices[valid_items]
        self.mean_speed = self.mean_speed[valid_items]


def get_video_detector(parameters_model: str, parameters_video: str, opt=None):
    detector_yolo = vd.yolor_detector(**parameters_model)
    analizer = vd.video_analizer(detector_yolo, parameters_model['number_classes'], config_video=parameters_video)
    return analizer


def main(opt):
    with open(opt.config_detector, errors='ignore') as f:
        parameters_model = yaml.safe_load(f)  # load hyps dict

    with open(opt.config_video, errors='ignore') as f:
        parameters_video = yaml.safe_load(f)  # load hyps dict
    analizer = get_video_detector(parameters_model, parameters_video, opt)
    tracker = video_tracker(analizer, parameters_video)
    if opt.save_video:
        tracker.save_tracking_video_records(opt.path_video, opt.save_file)
    else:
        tracker.save_tracking_records(opt.path_video, opt.save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='video_tracking.py',
                                     description='This program process a video file and saves the count of '
                                                 'objects detected.')

    parser.add_argument('--path-video', type=str, help='path to video file')
    parser.add_argument('--model', default='yolor')

    parser.add_argument('--save-file', type=str, default='count.csv', help='Path to save video file.')

    parser.add_argument('--config-detector', type=str,
                        help='path to config file for detector settings (example data/config_yolor.yaml)')
    parser.add_argument('--config-video', type=str,
                        help='path to config file for handling video settings.')

    parser.add_argument('--ext', type=str, default='.png', help='extensions name')
    parser.add_argument('--rotate', action='store_true', help='Rotate cut if True.')

    parser.add_argument('--shape', nargs='+', type=int, default=[1024, 1024],
                        help='shape of images to extract from frame video ')
    parser.add_argument('--split', nargs='+', type=int, default=[2, 1],
                        help='Way to split the image to not loss format')

    parser.add_argument('--save-video', action='store_true', help='create video file with tracking registers.')

    opt = parser.parse_args()
    np.random.seed(60006)

    path_videos = '/home/luis/2022/cherry_2022/visita7_18_11_22/videos/'
    name_videos = [name for name in sorted(os.listdir(path_videos))]
    name_vid = name_videos[3]  # short video
    # name_vid = name_videos[6] # long video

    opt.path_video = os.path.join(path_videos, name_vid)

    opt.config_detector = 'data/config_yolor.yaml'
    opt.config_video = 'data/processing.yaml'
    opt.path_to_save = os.path.join('/home/luis/2022/cherry_2022/experiments/visit7', name_vid.replace('.MOV', '.avi'))
    opt.save_video = False
    # opt.path_to_save = opt.path_video + '_2'
    # opt.ext = '.JPG'
    opt.save_file = 'count2.csv'
    main(opt)
