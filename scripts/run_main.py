#!/usr/bin/python3

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Input, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from keras import backend as K

import numpy as np
import cv2

import sys
import argparse

cv2.ocl.setUseOpenCL(False)

class PModel:

    def __init__(self, modelFile, blockNumber = 4, sizeRange = None):

        self.config = tf.compat.v1.ConfigProto()

        self.config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.config.gpu_options.allow_growth                    = True

        self.blockNumber = blockNumber
        self.sizeRange   = sizeRange
        self.modelFile    = modelFile


    def __unetModelBlocks(self, blockNumber = 4, filterNumber = 16):

        inputs        = Input((None, None, 3))
        blockFeatures = []

        x = inputs

        for i in range(blockNumber):

            fnCur = filterNumber*(2**(i))
            conv1 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
            conv1 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            blockFeatures.append(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            x = pool1

        fnCur = filterNumber*(2**(blockNumber))
        conv3 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        conv3 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        drop3 = Dropout(0.5)(conv3)
        x = drop3
        for i in range(blockNumber):
            fnCur = filterNumber*(2**(blockNumber - i - 1))
            up8 = Conv2D(fnCur, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(x))
            merge8 = concatenate([blockFeatures.pop(), up8], axis=3)

            conv8 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(fnCur, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)


            x = conv8

        conv10 = Conv2D(1, 1, activation='sigmoid')(x)
        model = Model(inputs, conv10)

        return inputs,conv10,model

    def __splitAndPredict(self,p_x,p_y,image):

        with tf.compat.v1.Session(config = self.config):

            _, _, self.model = self.__unetModelBlocks(blockNumber = self.blockNumber)
            self.model.load_weights(self.modelFile)

            partsX = p_x*2 - 1
            partsY = p_y*2 - 1

            XParts      = [((image.shape[1]//p_x)*x//2,(image.shape[1]//(p_x))*(x+2)//2) for x in range(partsX)]
            YParts      = [((image.shape[0]//p_y)*x//2,(image.shape[0]//(p_y))*(x+2)//2) for x in range(partsY)]
            total_image = np.zeros((image.shape[0],image.shape[1]))

            for y1,y2 in YParts:
                for x1,x2 in XParts:

                    t_img = np.float32(image[y1:y2,x1:x2,:3])
                    in_image = np.expand_dims(t_img,axis=0)

                    print("in_image {}:{} {}:{} -> {}".format(y1, y2, x1, x2, in_image.shape))

                    t_img = self.model.predict(in_image, batch_size=1)
                    total_image[y1:y2,x1:x2] = np.maximum(t_img[0,...,0], total_image[y1:y2,x1:x2])

        K.clear_session()

        return total_image

    def __non_max_suppression_reverse(self, msk, filter_size):

        orig_mask = msk.copy()
        kernel = np.ones((filter_size,filter_size), dtype=np.uint8)
        kernel[filter_size//2, filter_size//2] = 0
        dilated_mask = cv2.dilate(orig_mask, kernel, iterations=1)
        local_max_mask = (dilated_mask < orig_mask).astype(np.uint8)
        local_plateau_mask = (dilated_mask == orig_mask).astype(np.uint8)
        local_plateau_mask[dilated_mask == 0] = 0

        _, _, _, centroids = cv2.connectedComponentsWithStats(local_plateau_mask)
        centroids = centroids[~np.isnan(centroids).any(axis=1)]
        centroids = centroids.astype(int)
        ret_mask = np.zeros((orig_mask.shape), dtype=orig_mask.dtype)

        ret_mask[centroids[:, 1], centroids[:, 0]] = 1
        ret_mask[local_max_mask > 0] = 1

        return ret_mask

    def __generate_detections_from_mask(self, mask, threshold=127):

        uint_mask   = np.array(mask * 255).astype(np.uint8)
        ret, thresh = cv2.threshold(uint_mask, threshold, 255, 0)
        
        dist_transform = cv2.distanceTransform(thresh.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        skeleton = self.__non_max_suppression_reverse(dist_transform, 11)
        row_ind, col_ind = np.where(skeleton > 0)
        boxes = np.array(
            [[x[1]-dist_transform[x[0],x[1]]*1, x[0]-dist_transform[x[0],x[1]]*1, 2 * int(dist_transform[x[0], x[1]]), 2 * int(dist_transform[x[0], x[1]])] for x in
             zip(row_ind, col_ind)])

        return boxes

    def __non_max_sup_boxes(self, boxes, overlapThresh=0.5):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:

            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")

    def predict(self, image):

        D = 256

        xSmall = ySmall = False
        if image.shape[0] < D:
            print("Too small x axis")
            xSmall = True
        if image.shape[1] < D:
            print("Too small y axis")
            ySmall = True

        image = cv2.resize(image, (D if ySmall else image.shape[1], D if xSmall else image.shape[0]), interpolation=cv2.INTER_AREA) if any([xSmall, ySmall]) else image

        print("Prediction request: image shape {}".format(image.shape))
        
        origImage = image[..., ::-1].astype(np.float32)/255.0
        origShape = origImage.shape
        refShape  = origImage.shape

        partsY       = refShape[0] // D
        partsX       = refShape[1] // D
        divisorPower = 1

        print("PartsY {} PartsX {}".format(partsY, partsX))

        while partsY % (2 ** divisorPower) == 0 and partsX % (2 ** divisorPower) == 0:
            divisorPower += 1
        divisorRequired = (2 ** (self.blockNumber - divisorPower + 2))

        print("DivisorRequired {}".format(divisorRequired))

        refShape = [refShape[0] // (partsY * divisorRequired) * (partsY * divisorRequired),
                    refShape[1] // (partsX * divisorRequired) * (partsX * divisorRequired),
                    refShape[2]]

        print("RefShape {}".format(refShape))

        origImage = cv2.resize(origImage, (refShape[1], refShape[0]), interpolation=cv2.INTER_AREA)

        print("OrigImage shape {}".format(origImage.shape))

        maskResult = self.__splitAndPredict(partsX, partsY, origImage)
        maskResult = cv2.resize(maskResult, (maskResult.shape[1]//2, maskResult.shape[0]//2), interpolation=cv2.INTER_AREA)
        maskResult = maskResult / np.max(maskResult)
        tBoxes     = self.__generate_detections_from_mask(maskResult, threshold=50)

        print("Found {} boxes".format(tBoxes))

        if(len(tBoxes) != 0):

            tBoxes[:,0] = tBoxes[:,0] / maskResult.shape[0] * origShape[0]
            tBoxes[:,1] = tBoxes[:,1] / maskResult.shape[1] * origShape[1]
            tBoxes[:,2] = tBoxes[:,2] / maskResult.shape[0] * origShape[0]
            tBoxes[:,3] = tBoxes[:,3] / maskResult.shape[1] * origShape[1]

            tBoxes = np.array(list(filter(lambda x: (x[2] * x[3] > 0) and (x[0] > 0) and (x[1] > 0), tBoxes)))
            if self.sizeRange != None:
                tBoxes = np.array(list(filter(lambda x: x[2] <  1.0*self.sizeRange[1] and x[3] <  1.0*self.sizeRange[1] and
                                                        x[2] >= 0.5*self.sizeRange[0] and x[3] >= 0.5*self.sizeRange[0] and
                                                        x[0] >  0.0 and x[1] > 0.0, tBoxes)))


            tBoxes[:,0] = tBoxes[:,0] + tBoxes[:,2]//2
            tBoxes[:,1] = tBoxes[:,1] + tBoxes[:,3]//2


            tBoxes = self.__non_max_sup_boxes(tBoxes, overlapThresh=0.7)

            for b in tBoxes:
                x = b[0]
                y = b[1]
                w = b[2]
                h = b[3]

                cv2.rectangle(image,(x-w//2,y-h//2),(x+w//2,y+h//2),(0,255,0),2)

            #cv2.putText(image, text = "Stevilo detektiranih polipov je {}".format(len(tBoxes)), org = (50,100), color = (0,255,0), thickness = 2)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import glob, os

class FolderProcessing:
    def __init__(self, detection_method, folder, img_ext, out_folder=None):
        self.detection_method = detection_method
        self.img_list = glob.iglob(os.path.join(folder,'*.' + img_ext))
        self.img_list = sorted(self.img_list)

        self.out_folder = out_folder

    def run(self):
        for img_filename in self.img_list:

            frame = cv2.imread(img_filename)
            frame = self.detection_method.predict(frame)

            if self.out_folder != None:
                cv2.imwrite(os.path.join(self.out_folder, os.path.basename(img_filename)),frame)
            else:
                import pylab as plt
                plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                plt.show(block=True)

def main(args):

    if args.image_folder is None:
        from echolib_wrapper import EcholibWrapper
        processer = lambda d: EcholibWrapper(d, args.out_channel, args.in_channel)
    else:
        processer = lambda d: FolderProcessing(d, args.image_folder, args.image_ext, args.out_folder)

    p = processer(PModel(modelFile=args.model))

    try:
        p.run()
    except Exception as ex:
        print(ex)
        pass


def parseArgs():
    parser = argparse.ArgumentParser(description='Poly Counting')
    parser.add_argument(
        '--model',
        dest='model',
        help='model model file',
        default="/opt/poco_model.hdf5",
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--image-folder',
        dest='image_folder',
        help='folder to images for processing (default: None)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--out-folder',
        dest='out_folder',
        help='folder to store output (default: None)',
        default=None,
        type=str
    )

    parser.add_argument(
        'out_channel', help='out channel for echolib', default=None
    )
    parser.add_argument(
        'in_channel', help='input channel for echolib', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    main(args)
