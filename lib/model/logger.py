import numpy as np
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

IDX2NAME = [
    'background', 'airplane', 'car', 'guitar', 'laptop', 'pistol', 'bag', 'chair', 'knife', 'motorbike', 'rocket', 
    'table', 'cap', 'earphone', 'lamp', 'mug', 'skateboard']

class SimulatedTrainingLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.save_step = 0

    def write(self, train_log, img, gt_box):
        self.save_step += 1
        self._add_scalars(self.save_step, train_log)
        self._add_images(self.save_step, img, gt_box)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_scalars(self, save_step, train_log):
      for key in train_log.keys():
        self.writer.add_scalars(key, {'train': train_log[key]}, save_step)

    def _add_images(self, save_step, img, dets):
        train_imgs = []
        for b in range(img.size(0)):
          train_img = img[b].permute(1, 2, 0).numpy()
          train_img = train_img[:, :, ::-1].copy()
          
          gt_box = dets[b] # [n, 5]
          for i in range(gt_box.size(0)):
            if gt_box[i, 4] == 0:
                continue
            bbox = tuple(int(np.round(x)) for x in gt_box[i, :4])
            cv2.rectangle(train_img, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            class_name = IDX2NAME[int(gt_box[i, 4])]
            cv2.putText(train_img, class_name, (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        2.0, (0, 0, 255), thickness=2)

          train_img = torch.from_numpy(train_img).permute(2,0,1)

          train_imgs += [train_img]
        train_imgs = torch.stack(train_imgs, 0)

        train_imgs = make_grid(train_imgs, nrow=1, normalize=True, scale_each=True, pad_value=1)
        self.writer.add_image('train', train_imgs)


class SimulatedTestingLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.save_step = 0

    def write(self, im, im_pred):
        self.save_step += 1
        # self._add_scalars(self.save_step, train_log)
        self._add_images(self.save_step, im, im_pred)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    # def _add_scalars(self, save_step, train_log):
    #   for key in train_log.keys():
    #     self.writer.add_scalars(key, {'train': train_log[key]}, save_step)

    def _add_images(self, save_step, im, im_pred):

        im_pred_grid = make_grid(im_pred, nrow=1, normalize=True, scale_each=True, pad_value=1)
        im_grid = make_grid(im, nrow=1, normalize=True, scale_each=True, pad_value=1)

        test_grid = torch.cat((im_grid, im_pred_grid), dim=-1)
        self.writer.add_image('test', test_grid)