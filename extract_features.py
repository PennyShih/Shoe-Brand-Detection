# import the necessary packages
from __future__ import print_function
from sklearn.feature_extraction.image import extract_patches_2d
import helpers
from hog import HOG
import dataset
from conf import Conf
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import cv2
import os
import xml.etree.ElementTree as ET

# construct the argument parser and parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

# load the configuration file
conf = Conf(args["conf"])

# initialize the HOG descriptor along with the list of data and labels
hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
          cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"])
data = []
labels = []

# grab the set of ground-truth images
# 使用 paths.list_images 可以自動抓取 jpg, png, jpeg 等所有圖片格式
trnPaths = list(paths.list_images(conf["image_dataset"]))

# 如果設定檔中有指定 percent_gt_images (例如只用 50% 資料)，這邊會進行抽樣
# 您的設定是 1.0 (100%)，所以這邊會全用
if conf["percent_gt_images"] < 1.0:
    trnPaths = random.sample(trnPaths, int(len(trnPaths) * conf["percent_gt_images"]))

print("[INFO] describing training ROIs (Positive Samples)...")

# setup the progress bar
widgets = ["Extracting: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(trnPaths), widgets=widgets).start()

# loop over the training paths
for (i, trnPath) in enumerate(trnPaths):
    # load the image, convert it to grayscale
    image = cv2.imread(trnPath)

    # 防呆：如果讀不到圖片(例如壞檔)就跳過
    if image is None:
        print(f"\n[WARNING] 無法讀取圖片: {trnPath}，跳過。")
        pbar.update(i)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # === [改良重點 1] 智慧檔名處理 ===
    # 不管副檔名是 .jpg 還是 .png，我們只取檔名部分
    # 例如: "dataset/nike/images/shoe_01.jpg" -> "shoe_01"
    filename = os.path.basename(trnPath)
    imageID = os.path.splitext(filename)[0]

    # === [改良重點 2] 檢查 XML 是否存在 (防呆機制) ===
    xmlPath = "{}/{}.xml".format(conf["image_annotations"], imageID)
    if not os.path.exists(xmlPath):
        # 這裡不報錯，只是靜默跳過，或是您可以取消註解下面這行來查看
        # print(f"\n[INFO] 找不到 XML: {xmlPath}，跳過這張圖。")
        pbar.update(i)
        continue

    # === [改良重點 3] 讀取 XML 內容 ===
    try:
        tree = ET.parse(xmlPath)
        root = tree.getroot()

        # 尋找 XML 中的 object
        # 注意：這裡假設一張圖只取「第一個」找到的 object
        # 如果您的圖裡同時有多個 Nike Logo，需要改寫成迴圈 (for obj in root.findall('object'):)
        obj = root.find('object')

        if obj is None:
            print(f"\n[WARNING] XML 裡找不到 <object> 標籤: {xmlPath}")
            pbar.update(i)
            continue

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))

        # === [改良重點 4] 座標邊界修正與 Padding ===
        # 加上 Padding (offset)
        pad = conf["offset"]
        ymin = max(0, ymin - pad)
        xmin = max(0, xmin - pad)
        ymax = min(image.shape[0], ymax + pad)
        xmax = min(image.shape[1], xmax + pad)

        # 切割圖片 (ROI)
        roi = image[ymin:ymax, xmin:xmax]

        # 再次檢查 ROI 是否有效 (避免寬高為 0)
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue

        # Resize 到 HOG 視窗大小 (例如 96x48)
        roi = cv2.resize(roi, tuple(conf["window_dim"]), interpolation=cv2.INTER_AREA)

        # define the list of ROIs that will be described
        # 根據設定決定要不要做「水平翻轉」(Flip) 來增加資料量
        rois = (roi, cv2.flip(roi, 1)) if conf["use_flip"] else (roi,)

        # loop over the ROIs
        for roi in rois:
            # extract features from the ROI and update the list of features and labels
            features = hog.describe(roi)
            data.append(features)
            labels.append(1)  # 1 代表正樣本 (是 Nike/Adidas)

    except Exception as e:
        print(f"\n[ERROR] 處理 XML 出錯 {xmlPath}: {e}")
        continue

    # update the progress bar
    pbar.update(i)

# grab the distraction image paths and reset the progress bar
pbar.finish()
dstPaths = list(paths.list_images(conf["image_distractions"]))
pbar = progressbar.ProgressBar(maxval=conf["num_distraction_images"], widgets=widgets).start()
print("[INFO] describing distraction ROIs (Negative Samples)...")

# loop over the desired number of distraction images
for i in np.arange(0, conf["num_distraction_images"]):
    # randomly select a distraction images, load it, convert it to grayscale, and
    # then extract random pathces from the image
    image = cv2.imread(random.choice(dstPaths))

    # 防呆
    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 隨機從背景圖中切出好幾塊 (Patches) 當作負樣本
    patches = extract_patches_2d(image, tuple(conf["window_dim"]),
                                 max_patches=conf["num_distractions_per_image"])

    # loop over the patches
    for patch in patches:
        # extract features from the patch, then update teh data and label list
        features = hog.describe(patch)
        data.append(features)
        labels.append(-1)  # -1 代表負樣本 (不是 Nike/Adidas)

    # update the progress bar
    pbar.update(i)

# dump the dataset to file
pbar.finish()
print("[INFO] dumping features and labels to file...")
# 確保 output 資料夾存在
if not os.path.exists(os.path.dirname(conf["features_path"])):
    os.makedirs(os.path.dirname(conf["features_path"]))

dataset.dump_dataset(data, labels, conf["features_path"], "features")