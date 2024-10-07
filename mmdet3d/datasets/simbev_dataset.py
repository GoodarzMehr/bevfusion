import mmcv
import torch

import numpy as np

from .pipelines import Compose

from pytorch3d.ops import box3d_overlap

from mmdet.datasets import DATASETS
from torch.utils.data import Dataset

from pyquaternion import Quaternion as Q

from ..core.bbox import LiDARInstance3DBoxes, get_box_type


CAM_NAME = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

OBJECT_CLASSES = {
    12: 'pedestrian',
    14: 'car',
    15: 'truck',
    16: 'bus',
    18: 'motorcycle',
    19: 'bicycle'
}


@DATASETS.register_module()
class SimBEVDataset(Dataset):
    '''
    This class serves as the API for experiments on a CARLA dataset generated
    by SimBEV.

    Attributes:
        dataset_root: root directory of the dataset.
        ann_file: annotation file of the dataset.
        classes: dataset map classes.
        modality: modality of the dataset.
        test_mode: whether the dataset is used for training or testing.
        load_interval: interval of data samples.
    '''

    def __init__(
        self,
        dataset_root,
        ann_file,
        object_classes=None,
        map_classes=None,
        pipeline=None,
        modality=None,
        test_mode=False,
        filter_empty_gt=True,
        with_velocity=True,
        use_valid_flag=False,
        load_interval=64,
        box_type_3d='LiDAR'
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.ann_file = ann_file
        self.object_classes = object_classes
        self.map_classes = map_classes
        self.modality = modality
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag
        self.load_interval = load_interval

        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.epoch = -1

        self.CLASSES = self.get_classes(object_classes)

        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        # self.CLASSES = self.get_classes(map_classes)

        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        if self.modality is None:
            self.modality = dict(use_camera=True, use_lidar=True)

        if not self.test_mode:
            self._set_group_flag()

    def set_epoch(self, epoch):
        self.epoch = epoch
        
        if hasattr(self, 'pipeline'):
            for transform in self.pipeline.transforms:
                if hasattr(transform, 'set_epoch'):
                    transform.set_epoch(epoch)
    
    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names
    
    def get_cat_ids(self, index):
        info = self.data_infos[index]

        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []

        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        
        return cat_ids
        # gt_seg_path = info['GT_SEG']

        # mmcv.check_file_exist(gt_seg_path)
        
        # gt_masks = np.load(gt_seg_path)['data'][:, 52:308, 52:308]

        # car_mask = gt_masks[1]
        # truck_mask = np.logical_or(gt_masks[2], gt_masks[3])
        # cyclist_mask = np.logical_or(gt_masks[4], gt_masks[5], gt_masks[6])
        # pedestrian_mask = gt_masks[7]

        # road_mask = np.logical_and(
        #     gt_masks[0],
        #     np.logical_not(np.logical_or.reduce((car_mask, truck_mask, cyclist_mask, pedestrian_mask)))
        # )

        # gt_mask = np.array([road_mask, car_mask, truck_mask, cyclist_mask, pedestrian_mask])

        # categories = gt_masks.any(axis=(1, 2))

        # return np.where(categories == True)[0].tolist()
    
    def load_annotations(self, ann_file):
        annotations = mmcv.load(ann_file)

        data_infos = []

        for key in annotations['data']:
            data_infos += annotations['data'][key]['scene_data']
        
        data_infos = data_infos[::self.load_interval]

        self.metadata = annotations['metadata']

        data_infos = self.load_gt_bboxes(data_infos)

        return data_infos
    
    def load_gt_bboxes(self, infos):
        for info in infos:
            gt_boxes = []
            gt_names = []
            gt_velocities = []
            
            num_lidar_pts = []
            num_radar_pts = []
            valid_flag = []

            gt_det_path = info['GT_DET']

            mmcv.check_file_exist(gt_det_path)

            gt_det = np.load(gt_det_path, allow_pickle=True)

            # Ego to global transformation.
            ego2global = np.eye(4).astype(np.float32)
            
            ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
            ego2global[:3, 3] = info['ego2global_translation']

            # Lidar to ego transformation.
            lidar2ego = np.eye(4).astype(np.float32)
            
            lidar2ego[:3, :3] = Q(self.metadata['LIDAR']['sensor2ego_rotation']).rotation_matrix
            lidar2ego[:3, 3] = self.metadata['LIDAR']['sensor2ego_translation']

            global2lidar = np.linalg.inv(ego2global @ lidar2ego)

            for det_object in gt_det:
                for tag in det_object['semantic_tags']:
                    if tag in OBJECT_CLASSES.keys():
                        global_bbox_corners = np.append(det_object['bounding_box'], np.ones((8, 1)), 1)
                        bbox_corners = (global2lidar @ global_bbox_corners.T)[:3].T

                        center = ((bbox_corners[0] + bbox_corners[7]) / 2).tolist()

                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[2]))
                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[4]))
                        center.append(np.linalg.norm(bbox_corners[0] - bbox_corners[1]))

                        diff = bbox_corners[0] - bbox_corners[2]
                        
                        gamma = np.arctan2(diff[1], diff[0])

                        center.append(-gamma)

                        gt_boxes.append(center)
                        gt_names.append(OBJECT_CLASSES[tag])
                        gt_velocities.append(det_object['linear_velocity'][:2])
                        
                        num_lidar_pts.append(det_object['num_lidar_pts'])
                        num_radar_pts.append(det_object['num_radar_pts'])
                        valid_flag.append(det_object['valid_flag'])

            info['gt_boxes'] = np.array(gt_boxes)
            info['gt_names'] = np.array(gt_names)
            info['gt_velocity'] = np.array(gt_velocities)

            info['num_lidar_pts'] = np.array(num_lidar_pts)
            info['num_radar_pts'] = np.array(num_radar_pts)
            info['valid_flag'] = np.array(valid_flag)

        return infos
    
    def get_data_info(self, index):
        info = self.data_infos[index]

        data = dict(
            scene = info['scene'],
            frame = info['frame'],
            timestamp = info['timestamp'],
            gt_seg_path = info['GT_SEG'],
            gt_det_path = info['GT_DET'],
            lidar_path = info['LIDAR']
        )

        # Ego to global transformation.
        ego2global = np.eye(4).astype(np.float32)
        
        ego2global[:3, :3] = Q(info['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = info['ego2global_translation']
        
        data['ego2global'] = ego2global

        # Lidar to ego transformation.
        lidar2ego = np.eye(4).astype(np.float32)
        
        lidar2ego[:3, :3] = Q(self.metadata['LIDAR']['sensor2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = self.metadata['LIDAR']['sensor2ego_translation']
        
        data['lidar2ego'] = lidar2ego

        if self.modality['use_camera']:
            data['image_paths'] = []
            data['camera_intrinsics'] = []
            data['camera2lidar'] = []
            data['lidar2camera'] = []
            data['lidar2image'] = []
            data['camera2ego'] = []

            for camera in CAM_NAME:
                data['image_paths'].append(info['RGB-' + camera])

                # Camera intrinsics.
                camera_intrinsics = np.eye(4).astype(np.float32)

                camera_intrinsics[:3, :3] = self.metadata['camera_intrinsics']
                
                data['camera_intrinsics'].append(camera_intrinsics)
                
                # Lidar to camera transformation.
                camera2lidar = np.eye(4).astype(np.float32)

                camera2lidar[:3, :3] = Q(self.metadata[camera]['sensor2lidar_rotation']).rotation_matrix
                camera2lidar[:3, 3] = self.metadata[camera]['sensor2lidar_translation']

                data['camera2lidar'].append(camera2lidar)

                lidar2camera = np.linalg.inv(camera2lidar)
                
                data['lidar2camera'].append(lidar2camera)

                # Lidar to image transformation.
                lidar2image = camera_intrinsics @ lidar2camera

                data['lidar2image'].append(lidar2image)

                # Camera to ego transformation.
                camera2ego = np.eye(4).astype(np.float32)

                camera2ego[:3, :3] = Q(self.metadata[camera]['sensor2ego_rotation']).rotation_matrix
                camera2ego[:3, 3] = self.metadata[camera]['sensor2ego_translation']

                data['camera2ego'].append(camera2ego)

        annotations = self.get_ann_info(index)
        data['ann_info'] = annotations
        
        return data
    
    def get_ann_info(self, index):
        info = self.data_infos[index]

        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]

        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]

            nan_mask = np.isnan(gt_velocity[:, 0])
            
            gt_velocity[nan_mask] = [0.0, 0.0]
            
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

        return anns_results
    
    def pre_pipeline(self, results):
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d
    
    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)

        if input_dict is None:
            return None
        
        self.pre_pipeline(input_dict)
        
        example = self.pipeline(input_dict)

        if self.filter_empty_gt and (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None

        return example
    
    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        
        self.pre_pipeline(input_dict)
        
        example = self.pipeline(input_dict)

        return example
    
    def evaluate_map(self, results):
        thresholds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        num_classes = len(self.map_classes)
        num_thresholds = len(thresholds)

        tp = torch.zeros(num_classes, num_thresholds)
        fp = torch.zeros(num_classes, num_thresholds)
        fn = torch.zeros(num_classes, num_thresholds)

        for result in results:
            pred = result['masks_bev']
            label = result['gt_masks_bev']

            pred = pred.detach().reshape(num_classes, -1)
            label = label.detach().bool().reshape(num_classes, -1)

            pred = pred[:, :, None] >= thresholds
            label = label[:, :, None]

            tp += (pred & label).sum(dim=1)
            fp += (pred & ~label).sum(dim=1)
            fn += (~pred & label).sum(dim=1)

        ious = tp / (tp + fp + fn + 1e-7)

        metrics = {}
        
        for index, name in enumerate(self.map_classes):
            metrics[f'map/{name}/iou@max'] = ious[index].max().item()
            
            for threshold, iou in zip(thresholds, ious[index]):
                metrics[f'map/{name}/iou@{threshold.item():.2f}'] = iou.item()
        
        metrics['map/mean/iou@max'] = ious.max(dim=1).values.mean().item()
        
        return metrics
    
    def evaluate(self, results, **kwargs):
        metrics = {}

        if 'masks_bev' in results[0]:
            metrics.update(self.evaluate_map(results))

        if 'boxes_3d' in results[0]:
            simbev_eval = SimBEVDetectionEval(results)

            metrics['mAP'] = simbev_eval.evaluate()
        
        # print(results[0])
        # print(results[0]['boxes_3d'].corners)
        # print(results[0]['boxes_3d'].gravity_center)
        # print(results[0]['gt_bboxes_3d'].corners)
        # print(results[0]['gt_bboxes_3d'].gravity_center)
        
        print(metrics)
        
        return metrics
    
    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, index):
        pool = np.where(self.flag == self.flag[index])[0]
        
        return np.random.choice(pool)
    
    def __getitem__(self, index):
        if self.test_mode:
            return self.prepare_test_data(index)

        while True:
            data = self.prepare_train_data(index)

            if data is None:
                index = self._rand_another(index)
                continue
            
            return data
    
    def __len__(self):
        return len(self.data_infos)


class SimBEVDetectionEval:
    def __init__(self, results, iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        self.results = results

        self.iou_thresholds = iou_thresholds

    def evaluate(self):
        num_classes = len(self.object_classes)
        num_thresholds = len(self.iou_thresholds)

        ap = torch.zeros((num_classes, num_thresholds))

        for j, threshold in enumerate(self.iou_thresholds):

            tps = {i: torch.empty((0, )) for i in range(num_classes)}
            fps = {i: torch.empty((0, )) for i in range(num_classes)}

            scores = {i: torch.empty((0, )) for i in range(num_classes)}

            num_gt_boxes = {i: 0 for i in range(num_classes)}

            for result in self.results:
                boxes_3d = result['boxes_3d']
                scores_3d = result['scores_3d']
                labels_3d = result['labels_3d']
                gt_boxes_3d = result['gt_bboxes_3d']
                gt_labels_3d = result['gt_labels_3d']

                if len(boxes_3d.tensor) > 0:
                    boxes_3d_corners = boxes_3d.corners
                else:
                    boxes_3d_corners = torch.empty((0, 8, 3))

                if len(gt_boxes_3d.tensor) > 0:
                    gt_boxes_3d_corners = gt_boxes_3d.corners
                else:
                    gt_boxes_3d_corners = torch.empty((0, 8, 3))
                
                for cls in range(num_classes):
                    pred_mask = labels_3d == cls
                    
                    gt_mask = gt_labels_3d == cls

                    pred_boxes = boxes_3d_corners[pred_mask]
                    pred_scores = scores_3d[pred_mask]
                    
                    gt_boxes = gt_boxes_3d_corners[gt_mask]

                    sorted_indices = torch.argsort(-pred_scores)

                    pred_boxes = pred_boxes[sorted_indices]
                    pred_scores = pred_scores[sorted_indices]

                    _, ious = box3d_overlap(pred_boxes, gt_boxes)

                    assigned_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)

                    tp = torch.zeros(len(pred_boxes))
                    fp = torch.zeros(len(pred_boxes))

                    for i, pred_box in enumerate(pred_boxes):
                        iou_max = 0
                        max_gt_idx = -1

                        for j, gt_box in enumerate(gt_boxes):
                            if not assigned_gt[j]:
                                iou = ious[i, j]
                                
                                if iou > iou_max:
                                    iou_max = iou
                                    max_gt_idx = j
                        
                        if iou_max >= threshold:
                            tp[i] = 1

                            assigned_gt[max_gt_idx] = True
                        else:
                            fp[i] = 1
                    
                    tps[cls] = torch.cat((tps[cls], tp))
                    fps[cls] = torch.cat((fps[cls], fp))

                    scores[cls] = torch.cat((scores[cls], pred_scores))

                    num_gt_boxes[cls] += len(gt_boxes)

            for cls in range(num_classes):
                sorted_indices = torch.argsort(-scores[cls])

                tps[cls] = tps[cls][sorted_indices]
                fps[cls] = fps[cls][sorted_indices]

                tps[cls] = torch.cumsum(tps[cls], dim=0)
                fps[cls] = torch.cumsum(fps[cls], dim=0)

                recalls = tps[cls] / num_gt_boxes[cls]
                precisions = tps[cls] / (tps[cls] + fps[cls])

                ap[cls, j] = torch.trapz(precisions, recalls)

        return ap