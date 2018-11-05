'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Contributor     :Paulo Miguel Almeida
Date Created    :20160623
Date Modified   :20181105
version         :0.3
usage           :python make_predictions_1.py
python_version  :3.6.x
'''

import glob
import cv2
import caffe
import numpy as np
from caffe.proto import caffe_pb2

caffe.set_mode_gpu()

# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
# Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/ubuntu/workspace/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/mean.binaryproto', 'rb') as f:
    data = f.read()
    mean_blob.ParseFromString(data)
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

#Read model architecture and trained model's weights
net = caffe.Net('/home/ubuntu/workspace/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt',
                '/home/ubuntu/workspace/deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffe_model_1_iter_5000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2, 0, 1))

'''
Making predicitions
'''
# Reading image paths
test_img_paths = [img_path for img_path in glob.glob("../input/test1/*jpg")]


# Prediction holder
class PredictionHolder:

    def __init__(self, id, output_rat, output_other):
        self.id = id
        self.output_rat = output_rat
        self.output_other = output_other


predictions = []

for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    # Flow variables
    img_name = img_path.split('/')[-1][:-4]
    pred = pred_probas[0]

    predictions += [PredictionHolder(img_name, pred[1], pred[0])]
    print(img_path)
    print(pred)
    print(pred_probas.argmax())
    print('-------')


for threshold in np.arange(0.1, 1.1, 0.1):
    path = "../caffe_models/caffe_model_1/model_evaluation_tr_{}_.csv".format(str(threshold).replace('.', '_'))

    count_tp = 0
    count_tn = 0
    count_fp = 0
    count_fn = 0

    with open(path, "w") as f:
        f.write("ID,LABEL,THRESHOLD,TP,TN,FP,FN\n")

        for prediction in predictions:

            output_val = 1 if prediction.output_rat >= threshold else 0
            true_positive = 1 if "rat" in prediction.id and output_val == 1 else 0
            true_negative = 1 if "other" in prediction.id and output_val == 0 else 0
            false_positive = 1 if "other" in prediction.id and output_val == 1 else 0
            false_negative = 1 if "rat" in prediction.id and output_val == 0 else 0

            count_tp += true_positive
            count_tn += true_negative
            count_fp += false_positive
            count_fn += false_negative

            f.write("{},{},{},{},{},{},{}\n".format(
                prediction.id,
                str(output_val),
                str(threshold),
                str(true_positive),
                str(true_negative),
                str(false_positive),
                str(false_negative)
            ))

        precision = (count_tp/(count_tp+count_fp))
        recall = (count_tp/(count_tp+count_fn))

        f.write(",,,,,,\n")
        f.write(",,,{},{},{},{}\n".format(count_tp, count_tn, count_fp, count_fn))
        f.write(",,,,,,\n")
        f.write("Precision,{:.8f},,,,,\n".format(precision))
        f.write("Recall,{:.8f},,,,,\n".format(recall))
        f.write("F-Measure,{:.8f},,,,,\n".format(1/(0.9*(1/precision)+(1-0.9)*(1/recall))))

    f.close()
