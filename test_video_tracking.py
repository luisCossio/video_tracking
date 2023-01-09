import unittest
import yaml
import numpy as np

import video_tracking as vt


class dummy_analyzer:
    def __init__(self, nunmber_classes: int, config_video: dict):
        self.nunmber_classes = nunmber_classes
        self.shape_frame = config_video['shape_img']

    def get_next_prediction(self, amount=0):
        if amount == 0:
            random_items = np.random.randint(0, 30)
        else:
            random_items = amount
        random_prediction = np.empty([random_items, 6])
        random_prediction[:, :2] = np.random.uniform(0, self.shape_frame[0], [random_items, 2])
        random_prediction[:, 2:4] = np.random.uniform(0, 100, [random_items, 2]) + random_prediction[:, :2]
        random_prediction[:, 5] = np.random.randint(0, 3, [random_items])
        return random_prediction

    def get_next_frame_prediction(self):
        return np.array([0], dtype=np.uint8), self.get_next_prediction()

    def start_video_predictions(self, path_video):
        """
        Method to signal start extracting predictions from a video frame

        :return:
        """
        self.video_not_over = True

    def is_not_over(self):
        return False


class continuous_analyzer:
    def __init__(self, nunmber_classes: int, config_video: dict, n_items=9):
        self.number_classes = nunmber_classes
        self.shape_frame = config_video['shape_img']
        self.n_objects = n_items
        self.positions = np.empty([self.n_objects, 4], dtype=np.float32)
        self.positions[:, :2] = np.random.uniform(0, 600, [self.n_objects, 2])
        self.positions[:, 2:4] = np.random.uniform(6, 90, [self.n_objects, 2]) + self.positions[:, :2]
        self.classes_permanent = np.random.uniform(0, 1, [self.n_objects, 2])
        self.classes_permanent[:, 1] = np.random.randint(0, self.number_classes, [self.n_objects])

    def get_random_prediction(self, random_items):
        random_prediction = np.empty([random_items, 6], dtype=np.float32)
        random_prediction[:, :2] = np.random.uniform(0, self.shape_frame[0], [random_items, 2])
        random_prediction[:, 2:4] = np.random.uniform(0, 100, [random_items, 2]) + random_prediction[:, :2]
        random_prediction[:, 5] = np.random.randint(0, 3, [random_items])
        random_prediction[:, 4] = np.random.uniform(0.5, 1.0, [random_items])
        return random_prediction

    def get_next_prediction(self, amount=0):
        if amount == 0:
            n_detections = np.random.randint(0, 30, dtype=np.int32)
        else:
            n_detections = amount
        random_prediction = self.get_random_prediction(n_detections)
        min_items = min(n_detections, self.n_objects)
        random_prediction[:min_items, :4] = self.positions[:min_items]
        random_prediction[:min_items, 4:] = self.classes_permanent[:min_items]
        # random_prediction[:min_items,:4] += np.random.uniform(-5, 5, [min_items, 4])
        self.move_next_position()
        return random_prediction

    def get_next_frame_prediction(self):
        return None, self.get_next_prediction()

    def start_video_predictions(self, path_video):
        """
        Method to signal start extracting predictions from a video frame

        :return:
        """
        self.video_not_over = True

    def move_next_position(self):
        self.positions[:, :4] += np.random.uniform(-3, 1, [self.n_objects, 4])

    def is_not_over(self):
        return False


class reader_analizer:
    def __init__(self, nunmber_classes: int, config_video: dict, path_file='records_detections.csv'):
        self.number_classes = nunmber_classes
        self.shape_frame = config_video['shape_img']
        self.detection_data = np.loadtxt(path_file)

        row_values = np.mean(self.detection_data, axis=1)
        indices_separation = np.where(row_values == 0)[0]
        self.detections = []
        prev_index = 0
        for current_index in indices_separation:
            self.detections += [self.detection_data[prev_index:current_index]]
            prev_index = current_index + 1

        self.max_frames = len(self.detections)
        self.video_not_over = False
        self.frame_count = 0

    def get_next_prediction(self):
        prediction = self.detections[self.frame_count]
        self.frame_count += 1
        if self.frame_count == self.max_frames:
            self.video_not_over = False
        return prediction

    def get_next_frame_prediction(self):
        return None, self.get_next_prediction()

    def start_video_predictions(self, path_video):
        """
        Method to signal start extracting predictions from a video frame

        :return:
        """
        self.video_not_over = True
        self.frame_count = 0

    def is_not_over(self):
        return self.video_not_over

    def get_n_frames(self):
        return self.max_frames


def get_dummy_analizer(parameters_model, parameters_video, opt=None):
    analizer = dummy_analyzer(parameters_model['number_classes'], config_video=parameters_video)
    return analizer


def calculate_distance(vec1, vect2):
    dist = np.sum(np.abs(vec1 - vect2))
    return dist


class test_tracking(unittest.TestCase):
    def setUp(self):
        config_detector = 'data/config_yolor.yaml'
        config_video = 'data/processing_test.yaml'
        with open(config_detector, errors='ignore') as f:
            parameters_model = yaml.safe_load(f)  # load hyps dict

        with open(config_video, errors='ignore') as f:
            parameters_video = yaml.safe_load(f)  # load hyps dict

        analizer = get_dummy_analizer(parameters_model, parameters_video)
        self.tracker = vt.video_tracker(analizer, parameters_video, absent_frame=parameters_video['skip_rate'])
        self.analyzer = analizer
        self.config = parameters_video
        # tracker.save_tracking_records(path_video, save_file)
        np.random.seed(3)

    def test_start_records1(self):
        n_detections = 1
        n_classes = self.tracker.n_classes
        dummy_prediction = self.create_random_detections(n_detections)
        classes_predictions = dummy_prediction[:, 5]
        self.tracker.start_records(dummy_prediction)

        for i in range(n_detections):
            self.assertEqual(0, self.tracker.records_visible[i, 0])
            self.assertEqual(0, self.tracker.records_visible[i, 1])
            self.assertEqual(1, self.tracker.records_visible[i, 2])
            self.assertEqual(0, self.tracker.records_visible[i, 3])
            array_classes = np.zeros([n_classes], dtype=np.int32)
            array_classes[int(classes_predictions[i])] = 1
            for j in range(n_classes):
                self.assertEqual(array_classes[j], self.tracker.records_visible[i, 4 + j])
        self.assertEqual(0, len(self.tracker.final_records))

    def test_start_records2(self):
        n_detections = 10
        n_classes = self.tracker.n_classes
        dummy_prediction = self.create_random_detections(n_detections)
        classes_predictions = dummy_prediction[:, 5]
        self.tracker.start_records(dummy_prediction)

        for i in range(n_detections):
            self.assertEqual(0, self.tracker.records_visible[i, 0])
            self.assertEqual(0, self.tracker.records_visible[i, 1])
            self.assertEqual(1, self.tracker.records_visible[i, 2])
            self.assertEqual(0, self.tracker.records_visible[i, 3])
            array_classes = np.zeros([n_classes], dtype=np.int32)
            array_classes[int(classes_predictions[i])] = 1
            for j in range(n_classes):
                self.assertEqual(array_classes[j], self.tracker.records_visible[i, 4 + j])
        self.assertEqual(0, len(self.tracker.final_records))

    def test_start_records3(self):
        n_detections = 0
        n_classes = self.tracker.n_classes
        dummy_prediction = self.create_random_detections(n_detections)
        classes_predictions = dummy_prediction[:, 5]
        self.tracker.start_records(dummy_prediction)

        for i in range(n_detections):
            self.assertEqual(0, self.tracker.records_visible[i, 0])
            self.assertEqual(0, self.tracker.records_visible[i, 1])
            self.assertEqual(1, self.tracker.records_visible[i, 2])
            self.assertEqual(0, self.tracker.records_visible[i, 3])
            array_classes = np.zeros([n_classes], dtype=np.int32)
            array_classes[int(classes_predictions[i])] = 1
            for j in range(n_classes):
                self.assertEqual(array_classes[j], self.tracker.records_visible[i, 4 + j])
        self.assertEqual(0, len(self.tracker.final_records))

    def create_random_detections(self, n_detections):
        dummy_prediction = np.empty([n_detections, 6], dtype=np.float32)
        dummy_prediction[:, :2] = np.random.uniform(0, 1000, [n_detections, 2])
        dummy_prediction[:, 2:4] = np.random.uniform(0, 100, [n_detections, 2]) + dummy_prediction[:, :2]
        dummy_prediction[:, 5] = np.random.randint(0, 3, [n_detections])
        dummy_prediction[:, 4] = np.random.uniform(0.5, 1, [n_detections])
        return dummy_prediction

    def test_get_update_kalman1(self):
        """
        Test if initial positions move appropriately
        :return: None
        """
        new_detections = self.analyzer.get_next_prediction()
        self.tracker.start_records(new_detections)
        mean_speed = np.array([1, 1, 1, 1], dtype=np.int32)  # speed is 1 pixel in each direction
        new_detections_positions = vt.format_detections(new_detections[:, :4], mean_speed)
        prior_locations = self.tracker.get_update_kalman(new_detections_positions, mean_speed)  # get kalman prediction

        n_objects = len(new_detections)
        for i in range(n_objects):
            dist = calculate_distance(new_detections_positions[i, :4] + 1, prior_locations[i, :4])
            self.assertGreater(1, dist)  # distance from the predicted location.

    def test_connect_locations1(self):
        """
        Basic behavior testing 1: exact same position
        :return:
        """
        n_items_valid = 6
        n_items_invalid1 = 7
        n_items_invalid2 = 10
        start_row = 3
        prior_locations = self.create_random_detections(n_items_valid + n_items_invalid1)
        new_detections_positions = self.create_random_detections(n_items_valid + n_items_invalid2)
        new_detections_positions[start_row:(start_row + n_items_valid)] = prior_locations[:n_items_valid]

        indices_records, indices_detections = self.tracker.connect_locations(prior_locations[:, :4],
                                                                             new_detections_positions[:, :4],
                                                                             0.5)  # compare

        self.assertEqual(len(indices_records), len(indices_detections))
        max_dist = 1
        for i in range(n_items_valid):
            dist = calculate_distance(prior_locations[indices_records[i]],
                                      new_detections_positions[indices_detections[i]])
            self.assertGreater(max_dist, dist)

    def test_connect_locations2(self):
        """
        Basic behavior testing 2: similar positions
        :return:
        """
        n_items_valid = 6
        n_items_invalid1 = 7
        n_items_invalid2 = 10
        start_row = 0
        prior_locations = self.create_random_detections(n_items_valid + n_items_invalid1)
        new_detections_positions = self.create_random_detections(n_items_valid + n_items_invalid2)
        new_detections_positions[start_row:(start_row + n_items_valid)] = prior_locations[:n_items_valid]
        new_detections_positions[:, :4] += np.random.uniform(-5, 5, [n_items_valid + n_items_invalid2, 4])

        indices_records, indices_detections = self.tracker.connect_locations(prior_locations[:, :4],
                                                                             new_detections_positions[:, :4],
                                                                             0.5)  # compare

        self.assertEqual(len(indices_records), len(indices_detections))
        max_dist = 15
        for i in range(n_items_valid):
            dist = calculate_distance(prior_locations[indices_records[i]],
                                      new_detections_positions[indices_detections[i]])

            self.assertGreater(max_dist, dist)

    def test_connect_locations3(self):
        """
        Basic behavior testing 3: No match
        :return:
        """
        n_items_valid = 0
        n_items_invalid1 = 7
        n_items_invalid2 = 10
        prior_locations = self.create_random_detections(n_items_valid + n_items_invalid1)
        new_detections_positions = self.create_random_detections(n_items_valid + n_items_invalid2)

        indices_records, indices_detections = self.tracker.connect_locations(prior_locations[:, :4],
                                                                             new_detections_positions[:, :4],
                                                                             0.5)  # compare
        self.assertEqual(len(indices_records), len(indices_detections))
        self.assertEqual(0, len(indices_detections))

    def test_update_positions1(self):
        """
        Test1: Basic behavior test. Position of updated locations is similar to prior and detection positions.
        :return:
        """
        n_items_valid = 6
        n_items_invalid1 = 7
        n_items_invalid2 = 10
        start_row = 0

        mean_speed = - np.ones([n_items_valid + n_items_invalid1, 4], dtype=np.float32)
        initial_position = vt.format_detections(self.create_random_detections(n_items_valid + n_items_invalid1),
                                                mean_speed)
        self.tracker.start_records(initial_position)

        prior_locations = self.tracker.get_update_kalman(initial_position, mean_speed)

        new_detections_positions = self.create_random_detections(n_items_valid + n_items_invalid2)
        new_detections_positions = vt.format_detections(new_detections_positions)
        new_detections_positions[start_row:(start_row + n_items_valid), :4] = prior_locations[:n_items_valid, :4]
        new_detections_positions[:, :4] += np.random.uniform(-3, 3, [n_items_valid + n_items_invalid2, 4])

        indices_records = np.arange(0, n_items_valid)
        indices_detections = np.arange(start_row, start_row + n_items_valid)
        indices_records_not_found = np.delete(np.arange(len(self.tracker.records_visible)), indices_records)

        self.tracker.update_positions(prior_locations, new_detections_positions, indices_records,
                                      indices_records_not_found, indices_detections)

        n_objects = len(indices_records)
        final_positions = self.tracker.position_visible
        final_states = self.tracker.records_visible
        final_speed = self.tracker.mean_speed
        final_covs = self.tracker.records_visible

        self.assertEqual(len(final_positions), len(final_states))
        self.assertEqual(len(final_speed), len(final_states))
        self.assertEqual(len(final_covs), len(final_states))

        for i in range(n_objects):
            dist = calculate_distance(final_positions[indices_records[i]],
                                      new_detections_positions[indices_detections[i]])

            self.assertGreater(15, dist)  # distance from the predicted location.

            dist = calculate_distance(prior_locations[indices_records[i]],
                                      new_detections_positions[indices_detections[i]])

            self.assertGreater(15, dist)  # distance from the predicted location.

        indices_records_not_found = np.delete(np.arange(len(self.tracker.records_visible)), indices_records)
        for i in indices_records_not_found:
            for j in range(8):
                self.assertEqual(prior_locations[i][j], self.tracker.position_visible[i][j])

    def test_update_positions2(self):
        """
        Test 2: no match behavior
        :return:
        """
        n_items_valid = 0
        n_items_invalid1 = 7
        n_items_invalid2 = 10

        mean_speed = - np.ones([n_items_valid + n_items_invalid1, 4], dtype=np.float32)
        initial_position = vt.format_detections(self.create_random_detections(n_items_valid + n_items_invalid1),
                                                mean_speed)

        self.tracker.start_records(initial_position)

        prior_locations = self.tracker.get_update_kalman(initial_position, mean_speed)

        new_detections_positions = self.create_random_detections(n_items_valid + n_items_invalid2)
        new_detections_positions = vt.format_detections(new_detections_positions)
        # new_detections_positions[:, :4] += np.random.uniform(-3, 3, [n_items_valid + n_items_invalid2, 4])

        indices_records = np.empty([0], dtype=np.int32)
        indices_detections = np.empty([0], dtype=np.int32)
        indices_records_not_found = np.arange(n_items_invalid1)

        self.tracker.update_positions(prior_locations, new_detections_positions, indices_records,
                                      indices_records_not_found, indices_detections)

        final_positions = self.tracker.position_visible
        n_items_after = len(final_positions)
        self.assertEqual(n_items_invalid1, n_items_after)  # no new items are added, since that happens later

        final_speed = self.tracker.mean_speed
        final_states = self.tracker.records_visible
        final_covs = self.tracker.records_visible

        self.assertEqual(len(final_positions), len(final_states))
        self.assertEqual(len(final_speed), len(final_states))
        self.assertEqual(len(final_covs), len(final_states))

    def test_add_new_objects(self):
        n_items_valid = 6  # n matches
        n_items_invalid1 = 7  # n non-matches in prior positions
        n_items_invalid2 = 10  # n non-matches in detections

        mean_speed = - np.ones([n_items_valid + n_items_invalid1, 4], dtype=np.float32)
        initial_detections = self.create_random_detections(n_items_valid + n_items_invalid1)
        initial_position = vt.format_detections(initial_detections,
                                                mean_speed)
        self.tracker.start_records(initial_detections)

        prior_locations = self.tracker.get_update_kalman(initial_position, mean_speed)

        new_detections_positions = self.create_random_detections(n_items_valid + n_items_invalid2)
        new_detections_positions = vt.format_detections(new_detections_positions)
        new_detections_positions[:(n_items_valid), :4] = prior_locations[:n_items_valid, :4]
        new_detections_positions[:, :4] += np.random.uniform(-3, 3, [n_items_valid + n_items_invalid2, 4])

        indices_records = np.arange(0, n_items_valid)
        indices_detections = np.arange(n_items_valid)
        indices_records_not_found = np.delete(np.arange(len(self.tracker.records_visible)), indices_records)

        self.tracker.update_positions(prior_locations, new_detections_positions, indices_records,
                                      indices_records_not_found, indices_detections)

        classes_data = np.random.uniform(0, 1, [n_items_valid + n_items_invalid2, 2])
        classes_data[:, 1] = np.round(classes_data[:, 1] * 2)
        no_matches_detections = np.arange(n_items_valid, len(classes_data))
        self.tracker.add_new_objects(new_detections_positions[no_matches_detections],
                                     state_detections=classes_data[no_matches_detections])

        final_positions = self.tracker.position_visible
        final_states = self.tracker.records_visible

        n_positions_start = len(initial_position)
        n_positions_final = len(final_positions)

        self.assertEqual(n_items_invalid1 + n_items_valid, n_positions_start)
        self.assertEqual(n_items_invalid1 + n_items_valid + n_items_invalid2, n_positions_final)

        upper_bound = 0.1
        n_classes = self.tracker.n_classes
        for i in range(n_positions_start, n_positions_final):
            dist = calculate_distance(final_positions[i],
                                      new_detections_positions[i - n_positions_start + n_items_valid])

            self.assertGreater(upper_bound, dist)  # distance from the predicted location.
            self.assertEqual(1, final_states[i, 0])
            self.assertEqual(1, final_states[i, 2])
            self.assertEqual(0, final_states[i, 3])

            class_array = np.zeros([n_classes], dtype=np.int32)
            class_array[int(classes_data[i - n_positions_start + n_items_valid, 1])] = 1
            for j in range(n_classes):
                self.assertEqual(class_array[j], final_states[i, j + 4])

        final_speed = self.tracker.mean_speed
        final_states = self.tracker.records_visible
        final_covs = self.tracker.records_visible

        self.assertEqual(len(final_positions), len(final_states))
        self.assertEqual(len(final_speed), len(final_states))
        self.assertEqual(len(final_covs), len(final_states))

    def test_update_state1(self):
        n_items_valid = 5
        n_items_invalid1 = 7
        n_items_invalid2 = 10

        mean_speed = - np.ones([n_items_valid + n_items_invalid1, 4], dtype=np.float32)
        initial_detections = self.create_random_detections(n_items_valid + n_items_invalid1)

        self.tracker.start_records(initial_detections)
        initial_positions = self.tracker.position_visible
        prior_locations = self.tracker.get_update_kalman(initial_positions, mean_speed)

        new_detections, new_detections_positions = self.get_similar_detection(n_items_invalid2, n_items_valid,
                                                                              prior_locations)

        indices_records = np.arange(0, n_items_valid)
        indices_detections = np.arange(0, n_items_valid)
        indices_records_not_found = np.arange(n_items_valid, n_items_invalid1 + n_items_valid)

        n_items_before = len(initial_positions)
        self.tracker.update_positions(prior_locations, new_detections_positions, indices_records,
                                      indices_records_not_found, indices_detections)

        # classes_data = np.random.uniform(0, 1, [n_items_valid + n_items_invalid2, 2])
        indices_detections_not_seen = np.arange(n_items_valid, n_items_valid + n_items_invalid2)
        classes_data = new_detections[:, 4:]

        self.tracker.update_states(classes_data, indices_records, indices_detections, indices_records_not_found)

        self.tracker.add_new_objects(new_detections_positions[indices_detections_not_seen],
                                     state_detections=classes_data[indices_detections_not_seen])

        records_visible = self.tracker.records_visible
        n_items_after = len(records_visible)
        self.assertEqual(n_items_valid + n_items_invalid1 + n_items_invalid2, len(records_visible))

        for i in range(n_items_valid):
            self.assertEqual(0, records_visible[i, 0])
            self.assertEqual(1, records_visible[i, 1])
            self.assertEqual(2, records_visible[i, 2])
            self.assertEqual(0, records_visible[i, 3])

        for i in range(n_items_valid, n_items_before):
            self.assertEqual(0, records_visible[i, 0])
            self.assertEqual(0, records_visible[i, 1])
            self.assertEqual(-1, records_visible[i, 2])
            self.assertEqual(0, records_visible[i, 3])

        for i in range(n_items_before, n_items_after):
            self.assertEqual(1, records_visible[i, 0])
            self.assertEqual(1, records_visible[i, 1])
            self.assertEqual(1, records_visible[i, 2])
            self.assertEqual(0, records_visible[i, 3])

        final_positions = self.tracker.position_visible
        final_speed = self.tracker.mean_speed
        final_states = self.tracker.records_visible
        final_covs = self.tracker.covariance_matrices
        self.assertEqual(len(final_positions), len(final_states))
        self.assertEqual(len(final_positions), len(final_speed))
        self.assertEqual(len(final_covs), len(final_speed))

    def test_update_state2(self):
        """
        Non matches test
        :return:
        """
        n_items_valid = 0
        n_items_invalid1 = 7
        n_items_invalid2 = 10

        mean_speed = - np.ones([n_items_valid + n_items_invalid1, 4], dtype=np.float32)
        initial_detections = self.create_random_detections(n_items_valid + n_items_invalid1)

        self.tracker.start_records(initial_detections)
        initial_positions = self.tracker.position_visible
        prior_locations = self.tracker.get_update_kalman(initial_positions, mean_speed)

        new_detections, new_detections_positions = self.get_similar_detection(n_items_invalid2, n_items_valid,
                                                                              prior_locations)

        indices_records = np.empty([0], dtype=np.int32)
        indices_detections = np.empty([0], dtype=np.int32)
        indices_records_not_found = np.arange(n_items_invalid1)

        n_items_before = len(initial_positions)
        self.tracker.update_positions(prior_locations, new_detections_positions, indices_records,
                                      indices_records_not_found, indices_detections)

        indices_detections_not_seen = np.arange(n_items_valid, n_items_valid + n_items_invalid2)
        classes_data = new_detections[:, 4:]

        self.tracker.update_states(classes_data, indices_records, indices_detections, indices_records_not_found)

        self.tracker.add_new_objects(new_detections_positions[indices_detections_not_seen],
                                     state_detections=classes_data[indices_detections_not_seen])

        records_visible = self.tracker.records_visible
        n_items_after = len(records_visible)
        self.assertEqual(n_items_valid + n_items_invalid1 + n_items_invalid2, len(records_visible))

        for i in range(n_items_valid, n_items_before):
            self.assertEqual(0, records_visible[i, 0])
            self.assertEqual(0, records_visible[i, 1])
            self.assertEqual(-1, records_visible[i, 2])
            self.assertEqual(0, records_visible[i, 3])

        for i in range(n_items_before, n_items_after):
            self.assertEqual(1, records_visible[i, 0])
            self.assertEqual(1, records_visible[i, 1])
            self.assertEqual(1, records_visible[i, 2])
            self.assertEqual(0, records_visible[i, 3])

    def test_update_state3(self):
        """
        Testing non-continuous matches
        :return:
        """
        n_items_valid = 4
        n_items_invalid1 = 10
        n_items_invalid2 = 11

        mean_speed = - np.ones([n_items_valid + n_items_invalid1, 4], dtype=np.float32)
        initial_detections = self.create_random_detections(n_items_valid + n_items_invalid1)

        self.tracker.start_records(initial_detections)
        initial_positions = self.tracker.position_visible
        prior_locations = self.tracker.get_update_kalman(initial_positions, mean_speed)

        new_detections = self.create_random_detections(n_items_valid + n_items_invalid2)
        new_detections_positions = vt.format_detections(new_detections)
        indices_detections = np.array([(1 + i * 2) for i in range(n_items_valid)], dtype=np.int32)
        indices_records = np.array([(i * 2) for i in range(n_items_valid)], dtype=np.int32)
        new_detections_positions[indices_detections, :4] = prior_locations[indices_records, :4]

        n_items_before = len(initial_positions)
        indices_records_not_found = np.delete(np.arange(n_items_before), indices_records)

        self.tracker.update_positions(prior_locations, new_detections_positions, indices_records,
                                      indices_records_not_found, indices_detections)

        indices_detections_not_seen = np.delete(np.arange(n_items_valid + n_items_invalid2), indices_detections)
        classes_data = new_detections[:, 4:]

        self.tracker.update_states(classes_data, indices_records, indices_detections, indices_records_not_found)

        self.tracker.add_new_objects(new_detections_positions[indices_detections_not_seen],
                                     state_detections=classes_data[indices_detections_not_seen])

        records_visible = self.tracker.records_visible
        n_items_after = len(records_visible)
        self.assertEqual(n_items_valid + n_items_invalid1 + n_items_invalid2, len(records_visible))

        for i in indices_records_not_found:  # not found objects, seen in the first iteration
            self.assertEqual(0, records_visible[i, 0])
            self.assertEqual(0, records_visible[i, 1])
            self.assertEqual(-1, records_visible[i, 2])
            self.assertEqual(0, records_visible[i, 3])

        for i in indices_records:  # matched objects
            self.assertEqual(0, records_visible[i, 0])
            self.assertEqual(1, records_visible[i, 1])
            self.assertEqual(2, records_visible[i, 2])
            self.assertEqual(0, records_visible[i, 3])

        for i in range(n_items_before, n_items_after):  # new detections
            self.assertEqual(1, records_visible[i, 0])
            self.assertEqual(1, records_visible[i, 1])
            self.assertEqual(1, records_visible[i, 2])
            self.assertEqual(0, records_visible[i, 3])

    def get_similar_detection(self, n_items_non_match, n_items_match, locations):
        """
        Method to generate a copy of certain locations among random detections

        :param n_items_non_match:
        :param n_items_match:
        :param locations: locations to copy from
        :return:
        """
        new_detections = self.create_random_detections(n_items_match + n_items_non_match)
        new_detections_positions = vt.format_detections(new_detections)
        new_detections_positions[:n_items_match, :4] = locations[:n_items_match, :4]
        new_detections_positions[:n_items_match, :4] += np.random.uniform(-4, 4, [n_items_match, 4])
        return new_detections, new_detections_positions

    def test_update_records1(self):
        """
        Testing basic functionalities
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker
        n_items_valid = 3
        n_items_invalid1 = 4
        n_items_invalid2 = 8

        self.analyzer = continuous_analyzer(self.config['n_classes'], self.config, n_items=n_items_valid)
        new_detections = self.analyzer.get_next_prediction(n_items_invalid1 + n_items_valid)
        self.tracker.start_records(new_detections)
        positions1 = self.tracker.position_visible.copy()
        states1 = self.tracker.records_visible
        n_items1 = len(positions1)
        self.assertEqual(n_items1, len(states1))
        for i in range(len(positions1)):
            self.assertEqual(0, states1[i, 0])
            self.assertEqual(0, states1[i, 1])
            self.assertEqual(1, states1[i, 2])
            self.assertEqual(0, states1[i, 3])

        new_detections = self.analyzer.get_next_prediction(n_items_invalid2 + n_items_valid)
        self.tracker.update_records(new_detections)
        positions2 = self.tracker.position_visible
        states2 = self.tracker.records_visible
        n_items2 = len(positions2)

        indices_records = np.arange(0, n_items_valid)
        indices_records_not_found = np.delete(np.arange(n_items_invalid1 + n_items_valid), indices_records)
        self.assertEqual(len(positions2), len(states2))

        for i in indices_records_not_found:  # not found objects, seen in the first iteration
            self.assertEqual(0, states2[i, 0])
            self.assertEqual(0, states2[i, 1])
            self.assertEqual(-1, states2[i, 2])
            self.assertEqual(0, states2[i, 3])

        for i in indices_records:  # matched objects
            self.assertEqual(0, states2[i, 0])
            self.assertEqual(1, states2[i, 1])
            self.assertEqual(2, states2[i, 2])
            self.assertEqual(0, states2[i, 3])

        for i in range(n_items1, n_items2):  # new detections
            self.assertEqual(1, states2[i, 0])
            self.assertEqual(1, states2[i, 1])
            self.assertEqual(1, states2[i, 2])
            self.assertEqual(0, states2[i, 3])

    def test_update_records2(self):
        """
        Testing basic functionalities, more detections in first frame than in second
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker
        n_items_valid = 3
        n_items_invalid1 = 12
        n_items_invalid2 = 2

        self.analyzer = continuous_analyzer(self.config['n_classes'], self.config, n_items=n_items_valid)
        new_detections = self.analyzer.get_next_prediction(n_items_invalid1 + n_items_valid)
        self.tracker.start_records(new_detections)
        positions1 = self.tracker.position_visible.copy()
        states1 = self.tracker.records_visible
        n_items1 = len(positions1)
        self.assertEqual(n_items1, len(states1))
        for i in range(len(positions1)):
            self.assertEqual(0, states1[i, 0])
            self.assertEqual(0, states1[i, 1])
            self.assertEqual(1, states1[i, 2])
            self.assertEqual(0, states1[i, 3])

        new_detections = self.analyzer.get_next_prediction(n_items_invalid2 + n_items_valid)
        self.tracker.update_records(new_detections)
        positions2 = self.tracker.position_visible
        states2 = self.tracker.records_visible
        n_items2 = len(positions2)

        indices_records = np.arange(0, n_items_valid)
        indices_records_not_found = np.delete(np.arange(n_items_invalid1 + n_items_valid), indices_records)
        self.assertEqual(len(positions2), len(states2))

        for i in indices_records_not_found:  # not found objects, seen in the first iteration
            self.assertEqual(0, states2[i, 0])
            self.assertEqual(0, states2[i, 1])
            self.assertEqual(-1, states2[i, 2])
            self.assertEqual(0, states2[i, 3])

        for i in indices_records:  # matched objects
            self.assertEqual(0, states2[i, 0])
            self.assertEqual(1, states2[i, 1])
            self.assertEqual(2, states2[i, 2])
            self.assertEqual(0, states2[i, 3])

        for i in range(n_items1, n_items2):  # new detections
            self.assertEqual(1, states2[i, 0])
            self.assertEqual(1, states2[i, 1])
            self.assertEqual(1, states2[i, 2])
            self.assertEqual(0, states2[i, 3])

    def test_update_records3(self):
        """
        Testing basic functionalities, a lot of detections
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker
        n_items_valid = 3
        n_items_invalid1 = 69
        n_items_invalid2 = 67

        self.analyzer = continuous_analyzer(self.config['n_classes'], self.config, n_items=n_items_valid)
        new_detections = self.analyzer.get_next_prediction(n_items_invalid1 + n_items_valid)
        self.tracker.start_records(new_detections)
        positions1 = self.tracker.position_visible.copy()
        states1 = self.tracker.records_visible
        n_items1 = len(positions1)
        self.assertEqual(n_items1, len(states1))
        for i in range(len(positions1)):
            self.assertEqual(0, states1[i, 0])
            self.assertEqual(0, states1[i, 1])
            self.assertEqual(1, states1[i, 2])
            self.assertEqual(0, states1[i, 3])

        new_detections = self.analyzer.get_next_prediction(n_items_invalid2 + n_items_valid)
        self.tracker.update_records(new_detections)

        positions2 = self.tracker.position_visible
        states2 = self.tracker.records_visible
        n_items2 = len(positions2)

        indices_records = np.arange(0, n_items_valid)  # there are other matches, but these are definitively matches
        self.assertEqual(len(positions2), len(states2))

        for i in indices_records:  # matched objects
            self.assertEqual(0, states2[i, 0])
            self.assertEqual(1, states2[i, 1])
            self.assertEqual(2, states2[i, 2])
            self.assertEqual(0, states2[i, 3])

        for i in range(n_items1, n_items2):  # new detections
            self.assertEqual(1, states2[i, 0])
            self.assertEqual(1, states2[i, 1])
            self.assertEqual(1, states2[i, 2])
            self.assertEqual(0, states2[i, 3])

        final_positions = self.tracker.position_visible
        final_speed = self.tracker.mean_speed
        final_states = self.tracker.records_visible

        self.assertEqual(len(final_positions), len(final_states))
        self.assertEqual(len(final_positions), len(final_speed))

    def test_update_records4(self):
        """
        Testing basic functionalities: Only matches in each frame
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker
        n_items_valid = 3
        n_items_invalid1 = 0
        n_items_invalid2 = 0

        self.analyzer = continuous_analyzer(self.config['n_classes'], self.config, n_items=n_items_valid)
        new_detections = self.analyzer.get_next_prediction(n_items_invalid1 + n_items_valid)
        self.tracker.start_records(new_detections)

        indices_records = np.arange(n_items_valid)
        for i in range(4):  # necessary to define valid_ locations
            new_detections = self.analyzer.get_next_prediction(n_items_invalid2 + n_items_valid)
            self.tracker.update_records(new_detections)

            states = self.tracker.records_visible
            for j in indices_records:
                self.assertEqual(0, states[j, 0])
                self.assertEqual(i + 1, states[j, 1])
                self.assertEqual(i + 2, states[j, 2])

        for j in indices_records:
            self.assertEqual(1, states[j, 3])

    def test_update_records5(self):
        """
        Testing basic functionalities: no matches
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker
        n_items_valid = 0
        n_items_invalid1 = 3
        n_items_invalid2 = 3

        self.analyzer = dummy_analyzer(self.config['n_classes'], self.config)
        new_detections = self.analyzer.get_next_prediction(n_items_valid + n_items_invalid1)
        self.tracker.start_records(new_detections)

        n_iterations = 4
        for i in range(n_iterations):  # necessary to define valid_ locations
            new_detections = self.analyzer.get_next_prediction(n_items_valid + n_items_invalid2)
            self.tracker.update_records(new_detections)

            states = self.tracker.records_visible
            positions = self.tracker.position_visible
            self.assertEqual(len(states), len(positions))

            for j in range(n_items_invalid2):
                self.assertEqual(i + 1, states[-(j + 1), 0])
                self.assertEqual(i + 1, states[-(j + 1), 1])
                self.assertEqual(1, states[- (j + 1), 2])

        states = self.tracker.records_visible

        for i in range(1, n_iterations):
            for j in range(n_items_invalid2):
                self.assertEqual(i + 1, states[j + (i - 1) * n_items_invalid2, 0])
                self.assertEqual(i + 1, states[j + (i - 1) * n_items_invalid2, 1])

                self.assertEqual(-3 + (i - 1) * 2, states[j + (i - 1) * n_items_invalid2, 2])
                self.assertEqual(0, states[j + (i - 1) * n_items_invalid2, 3])

    def test_update_records_test_data1(self):
        """
        Testing basic functionalities on recorded data
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker

        file_records = 'records_detection_testing1.cvs'
        self.analyzer = reader_analizer(self.config['n_classes'], self.config, path_file=file_records)
        new_detections = self.analyzer.get_next_prediction()
        self.tracker.start_records(new_detections)
        positions1 = self.tracker.position_visible.copy()
        states1 = self.tracker.records_visible
        n_pos1 = len(states1)
        n_states1 = len(positions1)
        self.assertEqual(n_pos1, n_states1)

        for i in range(n_pos1):
            self.assertEqual(0, states1[i, 0])
            self.assertEqual(0, states1[i, 1])
            self.assertEqual(1, states1[i, 2])
            self.assertEqual(0, states1[i, 3])

        n_frames = self.analyzer.get_n_frames()
        for i in range(n_frames - 1):
            new_detections = self.analyzer.get_next_prediction()
            self.tracker.update_records(new_detections)
            self.test_new_positions(n_pos1)

    def test_update_records_test_data2(self):
        """
        Testing basic functionalities on recorded data
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker

        file_records = 'records_detection_testing2.cvs'
        self.analyzer = reader_analizer(self.config['n_classes'], self.config, path_file=file_records)
        new_detections = self.analyzer.get_next_prediction()
        self.tracker.start_records(new_detections)
        positions1 = self.tracker.position_visible.copy()
        states1 = self.tracker.records_visible
        n_pos1 = len(states1)
        n_states1 = len(positions1)
        self.assertEqual(n_pos1, n_states1)
        for i in range(n_pos1):
            self.assertEqual(0, states1[i, 0])
            self.assertEqual(0, states1[i, 1])
            self.assertEqual(1, states1[i, 2])
            self.assertEqual(0, states1[i, 3])

        n_frames = self.analyzer.get_n_frames()
        for i in range(n_frames - 1):
            new_detections = self.analyzer.get_next_prediction()
            self.tracker.update_records(new_detections)
            self.test_new_positions(n_pos1)

    def test_update_records_test_data3(self):
        """
        Testing basic functionalities on recorded data
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker

        file_records = 'records_detection_testing3.cvs'
        self.analyzer = reader_analizer(self.config['n_classes'], self.config, path_file=file_records)
        new_detections = self.analyzer.get_next_prediction()
        self.tracker.start_records(new_detections)
        positions1 = self.tracker.position_visible.copy()
        states1 = self.tracker.records_visible
        n_pos1 = len(states1)
        n_states1 = len(positions1)
        self.assertEqual(n_pos1, n_states1)

        for i in range(n_pos1):
            self.assertEqual(0, states1[i, 0])
            self.assertEqual(0, states1[i, 1])
            self.assertEqual(1, states1[i, 2])
            self.assertEqual(0, states1[i, 3])

        n_frames = self.analyzer.get_n_frames()
        for i in range(n_frames - 1):
            new_detections = self.analyzer.get_next_prediction()
            self.tracker.update_records(new_detections)
            self.test_new_positions(n_pos1)

    def test_update_records_test_data4(self):
        """
        Testing basic functionalities on recorded data
        :return: None
        """
        tracker = vt.video_tracker(self.analyzer, self.config, absent_frame=self.config['skip_rate'])
        self.tracker = tracker

        file_records = 'records_detection_testing4.cvs'
        self.analyzer = reader_analizer(self.config['n_classes'], self.config, path_file=file_records)
        new_detections = self.analyzer.get_next_prediction()
        self.tracker.start_records(new_detections)
        positions1 = self.tracker.position_visible.copy()
        states1 = self.tracker.records_visible
        n_pos1 = len(states1)
        n_states1 = len(positions1)
        self.assertEqual(n_pos1, n_states1)

        for i in range(n_pos1):
            self.assertEqual(0, states1[i, 0])
            self.assertEqual(0, states1[i, 1])
            self.assertEqual(1, states1[i, 2])
            self.assertEqual(0, states1[i, 3])

        n_frames = self.analyzer.get_n_frames()
        current_detection = new_detections.copy()
        for i in range(n_frames - 1):
            new_detections = self.analyzer.get_next_prediction()
            self.tracker.update_records(new_detections)
            self.test_new_positions(len(current_detection))  # the amount of positions decreases at some point to lower than 15
            current_detection = new_detections.copy()


        empty_detection = np.empty([0])
        self.tracker.update_records(empty_detection)
        self.test_new_positions(len(current_detection))

    def test_new_positions(self, n_pos1):
        new_positions = self.tracker.position_visible
        new_states = self.tracker.records_visible
        new_speeds = self.tracker.mean_speed
        new_covariances = self.tracker.covariance_matrices
        n_pos2 = len(new_states)
        n_states2 = len(new_positions)
        n_speed2 = len(new_speeds)
        n_covs2 = len(new_covariances)
        self.assertGreater(n_pos2, n_pos1)
        self.assertEqual(n_states2, n_pos2)
        self.assertEqual(n_speed2, n_pos2)
        self.assertEqual(n_covs2, n_pos2)