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


# === 輔助函式 ===
def load_model(conf_path):
    conf = Conf(conf_path)
    model = pickle.loads(open(conf["classifier_path"], "rb").read())
    hog = HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
              cellsPerBlock=tuple(conf["cells_per_block"]), normalize=conf["normalize"], block_norm="L1")
    od = ObjectDetector(model, hog)
    return conf, od


# === 設定參數 ===
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="圖片資料夾路徑 (例如: dataset/new_input)")
args = vars(ap.parse_args())

# === 步驟 1: 載入模型 ===
print("[INFO] 正在載入 Nike 模型...")
nike_conf, nike_od = load_model("conf/nike.json")

print("[INFO] 正在載入 Adidas 模型...")
adidas_conf, adidas_od = load_model("conf/adidas.json")

# 抓取所有圖片
imagePaths = list(paths.list_images(args["input"]))
total_images = 0
correct_predictions = 0

# 用來記錄詳細數據
stats = {
    "Nike": {"total": 0, "correct": 0},
    "Adidas": {"total": 0, "correct": 0},
    "Unknown": 0  # 檔名看不出品牌的圖
}

print(f"\n[INFO] 開始批量測試 {len(imagePaths)} 張圖片...\n")
print(f"{'Filename':<30} | {'True Label':<10} | {'Pred Label':<10} | {'N-Box':<5} | {'A-Box':<5} | {'Result'}")
print("-" * 85)

# === 步驟 2: 逐張測試 (不顯示視窗，只計算數據) ===
for (i, imagePath) in enumerate(imagePaths):
    filename = os.path.basename(imagePath)

    # 1. 解析 Ground Truth (從檔名判斷)
    true_label = "Unknown"
    if "nike" in filename.lower():
        true_label = "Nike"
    elif "adidas" in filename.lower():
        true_label = "Adidas"

    # 如果檔名裡沒有 nike 或是 adidas，我們就跳過不計分，或者歸類為未知
    if true_label == "Unknown":
        stats["Unknown"] += 1
        continue

    total_images += 1
    stats[true_label]["total"] += 1

    # 2. 讀取與前處理
    image = cv2.imread(imagePath)
    if image is None:
        continue
    image = imutils.resize(image, width=min(400, image.shape[1]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Nike 偵測
    (boxes_n, probs_n) = nike_od.detect(gray, nike_conf["window_dim"], winStep=nike_conf["window_step"],
                                        pyramidScale=nike_conf["pyramid_scale"], minProb=nike_conf["min_probability"])
    pick_n = non_max_suppression(np.array(boxes_n), probs_n, nike_conf["overlap_thresh"])
    score_nike = len(pick_n)  # 以「框框數量」作為分數

    # 4. Adidas 偵測
    (boxes_a, probs_a) = adidas_od.detect(gray, adidas_conf["window_dim"], winStep=adidas_conf["window_step"],
                                          pyramidScale=adidas_conf["pyramid_scale"],
                                          minProb=adidas_conf["min_probability"])
    pick_a = non_max_suppression(np.array(boxes_a), probs_a, adidas_conf["overlap_thresh"])
    score_adidas = len(pick_a)  # 以「框框數量」作為分數

    # 5. 判斷預測結果 (Prediction)
    pred_label = "None"

    if score_nike > score_adidas:
        pred_label = "Nike"
    elif score_adidas > score_nike:
        pred_label = "Adidas"
    else:
        # 平手狀況 (例如都為 0，或各 1 個)
        if score_nike > 0:  # 都有抓到但數量一樣
            # 這裡可以進階比信心度，但目前先簡單判定為無法區分
            pred_label = "Adidas"
        else:
            pred_label = "None"

    # 6. 比對結果
    is_correct = (pred_label == true_label)
    result_str = "V" if is_correct else "X"

    if is_correct:
        correct_predictions += 1
        stats[true_label]["correct"] += 1

    # 7. 顯示單行進度
    print(f"{filename:<30} | {true_label:<10} | {pred_label:<10} | {score_nike:<5} | {score_adidas:<5} | {result_str}")

# === 步驟 3: 輸出最終報表 ===
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0

print("\n" + "=" * 40)
print("             最終測試報告             ")
print("=" * 40)
print(f"總處理張數 (有標註): {total_images}")
print(f"總正確張數        : {correct_predictions}")
print(f"總體準確率 (Accuracy): {accuracy:.2f}%")
print("-" * 40)
print("各類別表現:")
if stats["Nike"]["total"] > 0:
    nike_acc = (stats["Nike"]["correct"] / stats["Nike"]["total"]) * 100
    print(f"Nike   : {stats['Nike']['correct']}/{stats['Nike']['total']} ({nike_acc:.2f}%)")
else:
    print(f"Nike   : 無測試樣本")

if stats["Adidas"]["total"] > 0:
    adidas_acc = (stats['Adidas']['correct'] / stats['Adidas']['total']) * 100
    print(f"Adidas : {stats['Adidas']['correct']}/{stats['Adidas']['total']} ({adidas_acc:.2f}%)")
else:
    print(f"Adidas : 無測試樣本")
print("=" * 40)