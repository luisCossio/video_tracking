import argparse
import os
import cv2
import numpy as np
import torch
import time
import yaml

# from abc import ABCMeta, abstractmethod
from typing import Protocol

from models_resources.models_yolo.yolo import Model
from utils_models.utils_yolo.torch_utils import intersect_dicts
import video_utils as vu


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y




def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None,
                        agnostic=False, redundant=True):  # require redundant detections
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after

    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros(0, 6)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
    return output


# class detector(metaclass=ABCMeta):
#     @abstractmethod
#     def get_predictions(self):
#         """
#         Returns a normalize prediction in the form of a numpy array with colums (x1,y1,x2,y2,conf,class)
#         :return:
#         """
#         pass

class detector(Protocol):
    def get_predictions(self):
        """
        Returns a normalize prediction in the form of a numpy array with colums (x1,y1,x2,y2,conf,class)
        :return:
        """
        ...

class yolor_detector:
    def __init__(self, configuration_file:str, weigths:str, number_classes:int, shape=[1024, 1024], device='cuda:0',
                 threshold_conf=0.3, iou_thres=0.6, merge=True):
        self.threshold = threshold_conf
        self.iou_thres = iou_thres
        self.merge = merge
        self.shape = shape
        self.device = torch.device(device)
        ckpt = torch.load(weigths, map_location=self.device)  # load checkpoint
        self.model = Model(configuration_file, ch=3, nc=number_classes).to(self.device)  # create
        if hasattr(ckpt['model'], 'state_dict'):
            state_dict = intersect_dicts(ckpt['model'].state_dict(), self.model.state_dict())  # intersect
        else:
            state_dict = intersect_dicts(ckpt['model'], self.model.state_dict())  # intersect
        # state_dict = intersect_dicts(ckpt['model'], model.state_dict())  # intersect
        self.model.load_state_dict(state_dict, strict=True)  # load

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        if self.half:
            self.model.half()

        # Configure
        self.model.eval()
        img = torch.zeros((1, 3, shape[0], shape[1]), device=device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    def transform_to_yolor(self, img: np.ndarray):
        # img, ratio, pad = letterbox(img, self.shape, auto=False, scaleup=False)
        img = cv2.resize(img,self.shape)
        img_plot = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_tensor = torch.from_numpy(img_plot).to(self.device)
        img_tensor = img_tensor.half() if self.half else img_tensor.float()  # uint8 to fp16/32
        img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        img_tensor = img_tensor.view(1, 3, self.shape[0], self.shape[1])
        return img_tensor

    def get_predictions(self, img:np.ndarray):
        img_device = self.transform_to_yolor(img)
        with torch.no_grad():
            inf_out, train_out = self.model(img_device, augment=False)  # inference and training outputs
        output = non_max_suppression(inf_out, conf_thres=self.threshold, iou_thres=self.iou_thres, merge=self.merge)[0]
        output = output.detach().to('cpu').numpy()
        if len(output) > 0:
            self.scale_predictions(output)
        return output.reshape([-1,6])

    def scale_predictions(self, output):
        array_shape = np.repeat(np.array(self.shape), 2)
        output[:, :4] = output[:, :4] / array_shape
        # return output


class analizer(Protocol):
    def get_next_frame_prediction(self):
        """
        Method to return next prediction and frame
        :return:
        """
        ...

    def get_next_prediction(self) -> np.ndarray:
        """
        Method to get next prediction

        :return: prediction
        """
        ...


    def start_video_predictions(self, file_path:str) -> None:
        """
        Method to signal start extracting predictions from file
        :param file_path: path to file
        :return: None
        """
        ...

    def is_not_over(self) -> bool:
        """
        :return: Return True if no more frames are available
        """
        ...


class video_analizer:
    def __init__(self, detector_object: detector, n_classes: int, config_video:dict):
        self.detector = detector_object
        self.number_classes = n_classes
        self.config_video = config_video
        self.count = 0
        self.skip_rate = config_video['skip_rate']
        self.video_not_over = False

    def analize_video_frame(self, path_video: str, path_save_images='', ext ='.JPG'):
        """
        Generate images from video frames with the drawn locations of each object identified.

        :param path_video: path to video file
        :param path_save_video: path to  save video
        :param shape: shape of images to extract from video
        :param configs: namespace for configurations
        :return:
        """

        records = vu.frame_extractor(path_video, rotate=self.config_video['rotate'])
        records.start()
        colors_classes = self.config_video['color_classes']
        count = 0
        while True:
            frame = records.get_next()
            frame = vu.cut_image(frame, size=self.config_video['shape_img'], start=[0, 0])
            if frame is None:
                break
            # print("shape frame: ", frame.shape)
            predictions = self.detector.get_predictions(frame)
            predictions[:,:4] = rescale(predictions[:,:4],self.config_video['shape_img'])
            image_name = 'f{:d}{:s}'.format(count, ext)

            if path_save_images and count % 5 == 0:
                img_save = vu.draw_image(frame.copy(), predictions, colors=colors_classes,
                                      number_classes=self.number_classes)
                path_new_image = os.path.join(path_save_images, image_name)
                cv2.imwrite(path_new_image, cv2.cvtColor(img_save, cv2.COLOR_BGR2RGB))
            count += 1

    def get_next_prediction(self)->np.ndarray:
        """
        Method to get predictions from the current frame in the recording
        :return: predictions
        """
        frame = self.get_next_frame()
        if frame is None:
            self.video_not_over = False
        else:
            predictions = self.split_image_and_detect(frame, self.config_video['shape_img'], self.config_video['split'])
            for i in range(self.skip_rate - 1):
                _ = self.get_next_frame()
            if _ is None:
                self.video_not_over = False
            return predictions
        return None

    def get_next_frame_prediction(self):
        """
        Method to return a prediction and its frame.

        :return:
        """
        frame = self.get_next_frame()
        if frame is None:
            self.video_not_over = False
        else:
            predictions = self.split_image_and_detect(frame, self.config_video['shape_img'], self.config_video['split'])
            for i in range(self.skip_rate - 1):
                _ = self.get_next_frame()
            if _ is None:
                self.video_not_over = False
            return frame, predictions
        return None, None

    def build_video(self, path_video: str, path_save_video='',frame_rate=30):
        """
        Function to create videos from the frames and predictions generated from the detector.

        :param path_video: path to video file
        :param path_save_video: path to  save video
        :param shape: shape of images to extract from video
        :param configs: namespace for configurations
        :return: None
        """
        records = vu.frame_extractor(path_video, rotate=self.config_video['rotate'])
        records.start()
        colors_classes = self.config_video['color_classes']
        count = 0
        list_frames = []
        splits = self.config_video['split']
        shape = self.config_video['shape_img']

        while True:
            frame = records.get_next()
            if frame is None:
                break

            if count % self.skip_rate == 0:  # for 30 fps
                predictions = self.split_image_and_detect(frame, shape, splits)
                img_save = vu.draw_image(frame.copy(), predictions, colors=colors_classes,
                                      number_classes=self.number_classes)
                list_frames += [cv2.cvtColor(img_save, cv2.COLOR_BGR2RGB)]
            count += 1

        if len(list_frames) == 0:
            print("ERROR Invalid video file")
            raise ValueError("Please enter a valid video file:\n {:s} \n This one has 0 frames".format(path_video))
        width = list_frames[0].shape[1]
        height = list_frames[0].shape[0]
        size = (width, height)

        out = cv2.VideoWriter(path_save_video, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
        print("Building Video:\n {:s}".format(path_save_video))
        for i in range(len(list_frames)):
            out.write(list_frames[i])
        out.release()

    def split_image_and_detect(self, frame, shape, splits) -> np.ndarray:
        predictions = []
        cuts, locations = vu.get_cuts(frame, shape, splits)
        for i in range(splits[0]):
            # increase_height = locations[i][0]
            for j in range(splits[1]):
                increase_height = locations[i][j][0]
                increase_width = locations[i][j][1]
                predictions_cut = self.detector.get_predictions(cuts[i][j])
                if len(predictions_cut) > 0:
                    predictions_cut[:, :4] = rescale(predictions_cut[:, :4], shape)
                    increase = np.tile(np.array([increase_width, increase_height]), 2)
                    predictions_cut[:, :4] += increase
                    predictions += [predictions_cut]
        if len(predictions) > 0:
            predictions = np.concatenate(predictions, axis=0)
            # predictions[:, :4] = rescale(predictions[:, :4], self.config_video['shape_img'])
        else:
            predictions = np.empty([0,6])
        return predictions

    def start_video_predictions(self, path_video):
        """
        Method to signal start extracting predictions from a video frame

        :return:
        """
        self.video_not_over = True
        self.records = vu.frame_extractor(path_video, rotate=self.config_video['rotate'])
        self.records.start()

    def is_not_over(self):
        return self.video_not_over

    def get_next_frame(self):
        """
        Method for iterative extraction of the predictions.
        :return:
        """
        if self.video_not_over:
            frame = self.records.get_next()
            while self.count % self.skip_rate != 0:
                frame = self.records.get_next()
                self.count += 1

                if frame is None:
                    self.video_not_over = False
                    break
            return frame
        return None

def rescale(coordinates,shape):
    array_shape = np.flip(np.tile(np.array(shape), 2))
    coordinates = coordinates * array_shape
    return coordinates


def analize_video(path_original_video: str, path_to_save: str, parameters_model, parameters_video, opt):
    detector_yolo = yolor_detector(**parameters_model)
    analizer = video_analizer(detector_yolo, parameters_model['number_classes'], config_video=parameters_video)
    analizer.build_video(path_original_video, path_to_save,opt.frame_rate)



def main(opt):

    with open(opt.config_detector, errors='ignore') as f:
        parameters_model = yaml.safe_load(f)  # load hyps dict

    with open(opt.config_video, errors='ignore') as f:
        parameters_video = yaml.safe_load(f)  # load hyps dict
    # analize_folder(opt.path_video, opt.path_to_save, parameters_model, opt)
    analize_video(opt.path_video, opt.path_to_save, parameters_model, parameters_video, opt)


#

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='video_detector.py',
                                     description='This program process a video file and saves a new one with the '
                                                 'detections of a given detector')

    parser.add_argument('--path-video', type=str, help='path to video file')
    parser.add_argument('--model', default='yolor')

    parser.add_argument('--path-to-save', type=str, default='', help='Path to save video file.')

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
    parser.add_argument('--frame-rate', type=int, default=30, help='extensions name')
    opt = parser.parse_args()
    np.random.seed(60006)

    # opt.path_video = '/home/luis/2022/cherry_2022/experiments/visit3/hilera88_arbol47'
    path_videos = '/home/luis/2022/cherry_2022/visita7_18_11_22/videos/'
    name_videos = [name for name in sorted(os.listdir(path_videos))]
    name_vid = name_videos[3]
    print(name_vid)
    opt.path_video = os.path.join(path_videos, name_vid)

    opt.config_detector = 'data/config_yolor.yaml'
    opt.config_video = 'data/processing.yaml'
    path_save_vid = '/home/luis/2022/cherry_2022/experiments/visit9'
    if not os.path.isdir(path_save_vid):
        os.mkdir(path_save_vid)
    opt.path_to_save = os.path.join(path_save_vid, name_vid.replace('.MOV','.avi'))
    opt.rotate = True
    opt.shape = [720, 720]
    opt.split = [2, 1]
    opt.frame_rate = 10
    # opt.path_to_save = opt.path_video + '_2'
    # opt.ext = '.JPG'
    main(opt)
