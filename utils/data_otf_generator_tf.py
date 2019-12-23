from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import glob
from tensorflow.data import Dataset
import tensorflow as tf
import os

class CropGenerator():
    def random_crop(self, img, random_crop_size, pos=None):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3 or img.shape[2] == 1
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        if pos is not None:
            x = pos[0]
            y = pos[1]
        else:
            x = np.random.randint(0, width - dx + 1)
            y = np.random.randint(0, height - dy + 1)
        return img[y:(y + dy), x:(x + dx), :]

    def __call__(self):
        '''
        Take as input a Keras ImageGen (Iterator) and generate random
        crops from the image batches generated by the original iterator
        '''
        num_crops = (self.target_shape[0]//self.crop_length+1) * (self.target_shape[1]//self.crop_length+1)
        while True:
            batch_x, batch_y = next(self.batches)
            batch_crops = np.zeros((batch_x.shape[0], self.crop_length, self.crop_length, 3))
            batch_y_crops = np.zeros((batch_x.shape[0], self.crop_length, self.crop_length, 1))
            for i in range(batch_x.shape[0]):
                height, width = batch_x[i].shape[0], batch_x[i].shape[1]
                for rep in range(num_crops):
                    x = np.random.randint(0, width - self.crop_length + 1)
                    y = np.random.randint(0, height - self.crop_length + 1)
                    batch_crops[i] = self.random_crop(batch_x[i], (self.crop_length, self.crop_length), pos=(x, y))
                    batch_y_crops[i] = self.random_crop(batch_y[i], (self.crop_length, self.crop_length), pos=(x, y))
                    yield (batch_crops, batch_y_crops)

    def __init__(self, dir_name, target_shape):
        self.crop_length = 512
        data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=0.0,
                             width_shift_range=0.0,
                             height_shift_range=0.0,
                             zoom_range=0.2,
                             rescale=1.0/255,
                             horizontal_flip=True,
                             vertical_flip=True
                             )
        self.dir_name = dir_name
        self.seed = np.random.randint(0,100)
        self.target_shape = target_shape

        image_datagen = ImageDataGenerator(**data_gen_args)
        image_generator = image_datagen.flow_from_directory(self.dir_name + '/training_data/images/', class_mode=None, shuffle=True,target_size = target_shape, batch_size=1, seed=self.seed)
        #mask_datagen = ImageDataGenerator(**data_gen_args)
        data_gen_args['featurewise_center'] = False
        data_gen_args['featurewise_std_normalization'] = False
        data_gen_args['rescale'] = 1.0/255
        mask_datagen = ImageDataGenerator(**data_gen_args)
        mask_generator = mask_datagen.flow_from_directory(self.dir_name + '/training_data/masks/', class_mode=None, shuffle=True, seed=self.seed,target_size = target_shape, batch_size=1, color_mode='grayscale')
        self.batches = zip(image_generator, mask_generator)


class dataset_generator():
    def __init__(self, dir_name, image_names, attributes):
        self.dir_name = dir_name
        self.image_names = image_names
        self.attributes = attributes

    def get_mask_image(self, image_path):
        base_name = tf.strings.split(image_path, os.path.sep, result_type='RaggedTensor')
        base_name = base_name[-1]

        mask_path = self.dir_name + "/training_data/masks/masks/" + base_name
        img = tf.io.read_file(mask_path)
        img = tf.image.decode_image(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def get_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def random_crop_and_pad_image_and_labels(self,image, mask, size):
        """Randomly crops `image` together with `labels`.

        Args:
          image: A Tensor with shape [D_1, ..., D_K, N]
          labels: A Tensor with shape [D_1, ..., D_K, M]
          size: A Tensor with shape [K] indicating the crop size.
        Returns:
          A tuple of (cropped_image, cropped_label).
        """
        combined = tf.concat([image, mask], axis=2)
        image_shape = tf.shape(image)
        combined_pad = tf.image.pad_to_bounding_box(
            combined, 0, 0,
            tf.maximum(size[0], image_shape[0]),
            tf.maximum(size[1], image_shape[1]))
        last_label_dim = tf.shape(mask)[-1]
        last_image_dim = tf.shape(image)[-1]
        combined_crop = tf.random_crop(
            combined_pad,
            size=tf.concat([size, [last_label_dim + last_image_dim]],
                           axis=0))
        return (combined_crop[:, :, :last_image_dim],
                combined_crop[:, :, last_image_dim:])

    def crop_image(self, image, mask):
        im_shape = image.shape
        w = im_shape[1]
        h = im_shape[2]
        x = tf.random.uniform([1],minval=0, maxval=w - 512 + 1, dtype=tf.dtypes.int32)
        y = tf.random.uniform([1],minval=0, maxval=h - 512 + 1, dtype=tf.dtypes.int32)
        # x = np.random.randint(0, w - 512 + 1)
        # y = np.random.randint(0, h - 512 + 1)
        image_crop = tf.image.crop_and_resize(image, boxes = [[x/w,y/h,(x+512.0)/w,(y+512.0)/h]], crop_size=[512,512])
        mask_crop = tf.image.crop_and_resize(mask, boxes = [[x/w,y/h,(x+512.0)/w,(y+512.0)/h]], crop_size=[512,512])
        return image_crop, mask_crop

    def get_image_mask_pair(self, file_path):
        image = self.get_image(file_path)
        mask = self.get_mask_image(file_path)
        image_crop, mask_crop = self.random_crop_and_pad_image_and_labels(image,mask,[512,512])
        return image_crop, mask_crop

    def prepare_train_generator(self):
        image_names = glob.glob(self.dir_name+"/training_data/images/images/*.jpg")
        image_names.extend(glob.glob(self.dir_name+"/training_data/images/images/*.png"))
        image_names.extend(glob.glob(self.dir_name+"/training_data/images/images/*.bmp"))
        sample_img = cv2.imread(image_names[0])
        target_shape = (sample_img.shape[0], sample_img.shape[1])


        crop_generator = CropGenerator(self.dir_name, target_shape)

        #image_dataset = tf.data.Dataset.list_files(self.dir_name + '/training_data/images/images/*')
        total_dataset = Dataset.range(1,8).interleave(lambda x: Dataset.from_generator(CropGenerator(self.dir_name, target_shape),output_types=(tf.float32, tf.float32)), cycle_length=8)
        total_dataset = total_dataset.shuffle(buffer_size=20)
        #total_dataset = total_dataset.cache("./data_cache.")
        total_dataset = total_dataset.repeat()
        total_dataset = total_dataset.prefetch(buffer_size=20)
        data_tf = total_dataset.make_one_shot_iterator().get_next()
        return data_tf, crop_generator()

    def prepare_predict_generator(self):
        image_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
        image_generator = image_datagen.flow_from_directory(self.dir_name + '/training_data/images/', class_mode=None,
                                                            shuffle=False)

        return image_generator
