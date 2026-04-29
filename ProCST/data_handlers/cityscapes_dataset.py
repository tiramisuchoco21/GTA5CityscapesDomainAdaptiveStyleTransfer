import numpy as np
from data_handlers.domain_adaptation_dataset import domainAdaptationDataSet
from core.functions import RGBImageToNumpy
import os.path as osp
from PIL import Image
from core.constants import RESIZE_SHAPE

class cityscapesDataSet(domainAdaptationDataSet):
    def __init__(self, root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=False, get_scales_pyramid=False, get_original_image=False):
        super(cityscapesDataSet, self).__init__(root, images_list_path, scale_factor, num_scales, curr_scale, set, get_image_label=get_image_label)
        self.get_scales_pyramid= get_scales_pyramid
        self.domain_resize = RESIZE_SHAPE['cityscapes']
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.get_original_image = get_original_image
    def __getitem__(self, index):
        import os
        name = self.img_ids[index]
        pure_name = os.path.basename(name)
        
        img_path = os.path.join(self.root, "images", pure_name)        
        if not os.path.exists(img_path):
            all_imgs = [f for f in os.listdir(os.path.join(self.root, "images")) if f.endswith(('.png', '.jpg'))]
            img_path = os.path.join(self.root, "images", all_imgs[index % len(all_imgs)])

        image = Image.open(img_path).convert('RGB')

        if not self.get_original_image:
            image = image.resize(self.domain_resize, Image.BICUBIC)

        scales_pyramid, label, label_copy = None, None, None
        
        if self.get_image_label:
            lb_pure_name = pure_name.replace("leftImg8bit", "gtFine_labelIds")
            label_path = os.path.join(self.root, "labels", lb_pure_name)            
            if not os.path.exists(label_path):
                label_path = os.path.join(self.root, "labels", pure_name)
                if not os.path.exists(label_path):
                    all_labels = [f for f in os.listdir(os.path.join(self.root, "labels")) if f.endswith('.png')]
                    label_path = os.path.join(self.root, "labels", all_labels[index % len(all_labels)])

            label = Image.open(label_path)
            
            if not self.get_original_image:
                label = label.resize(self.domain_resize, Image.NEAREST)
            
            label = np.asarray(label, np.float32)
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v

        scales_pyramid = None
        if self.get_scales_pyramid:
            scales_pyramid = self.GeneratePyramid(image)
        else:
            image = RGBImageToNumpy(image)

        if self.get_scales_pyramid and self.get_image_label:
            return scales_pyramid, label_copy.copy()
        elif self.get_scales_pyramid and not self.get_image_label:
            return scales_pyramid
        elif not self.get_scales_pyramid and self.get_image_label:
            return image.copy(), label_copy.copy()
        elif not self.get_scales_pyramid and not self.get_image_label:
            return image.copy()




