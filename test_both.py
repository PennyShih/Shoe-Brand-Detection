# import the necessary packages
from nms import non_max_suppression
from object_detection import ObjectDetector
from hog import HOG
from conf import Conf
from imutils import paths
import numpy as np
import imutils
import argparse
import pickle
import cv2
import os


# 輔助函式：用來載入設定檔與模型
def load_model(conf_path):
    conf = Conf(conf_path)
    model = pickle.loads(open(conf["classifier_path"], "rb").read())
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
              cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"], block_norm="L1")
    od = ObjectDetector(model, hog)
    return conf, od


# === 設定參數 (注意：這裡只有 -i，沒有 -c) ===
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="圖片資料夾路徑")
args = vars(ap.parse_args())

# === 步驟 1: 程式自動載入兩個模型 (寫死路徑) ===
print("[INFO] 正在載入 Nike 模型...")
# 請確保您的 json 檔案確實放在這個路徑
nike_conf, nike_od = load_model("conf/nike.json")

print("[INFO] 正在載入 Adidas 模型...")
adidas_conf, adidas_od = load_model("conf/adidas.json")

# 抓取所有圖片
imagePaths = list(paths.list_images(args["input"]))

# === 步驟 2: 逐張測試 ===
for (i, imagePath) in enumerate(imagePaths):
    filename = os.path.basename(imagePath)
    print(f"[INFO] ({i + 1}/{len(imagePaths)}) 正在分析: {filename}")

    # 讀取圖片
    image = cv2.imread(imagePath)
    if image is None:
        continue

    # 縮放圖片 (加速運算)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 複製一份原本的圖用來畫框
    output_image = image.copy()

    # --- [判斷 1] 呼叫 Nike 專家 ---
    (boxes_n, probs_n) = nike_od.detect(gray, nike_conf["window_dim"], winStep=nike_conf["window_step"],
                                        pyramidScale=nike_conf["pyramid_scale"], minProb=nike_conf["min_probability"])
    pick_n = non_max_suppression(np.array(boxes_n), probs_n, nike_conf["overlap_thresh"])

    # 畫 Nike 框框 (紅色: B=0, G=0, R=255)
    for (startX, startY, endX, endY) in pick_n:
        cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(output_image, "Nike", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # --- [判斷 2] 呼叫 Adidas 專家 ---
    (boxes_a, probs_a) = adidas_od.detect(gray, adidas_conf["window_dim"], winStep=adidas_conf["window_step"],
                                          pyramidScale=adidas_conf["pyramid_scale"],
                                          minProb=adidas_conf["min_probability"])
    pick_a = non_max_suppression(np.array(boxes_a), probs_a, adidas_conf["overlap_thresh"])

    # 畫 Adidas 框框 (藍色: B=255, G=0, R=0)
    for (startX, startY, endX, endY) in pick_a:
        cv2.rectangle(output_image, (startX, startY), (endX, endY), (255, 0, 0), 2)
        cv2.putText(output_image, "Adidas", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    # --- [結果統計] ---
    if len(pick_n) > 0 and len(pick_a) > 0:
        status = "Both Detected!"
    elif len(pick_n) > 0:
        status = "Found Nike"
    elif len(pick_a) > 0:
        status = "Found Adidas"
    else:
        status = "Nothing Found"

    # 顯示狀態在左上角
    cv2.putText(output_image, f"Status: {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 顯示圖片
    cv2.imshow("Result (Red=Nike, Blue=Adidas)", output_image)

    # 按 'q' 離開，按任意鍵下一張
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()