## HOG + SVM 鞋類商標自動偵測系統 (Shoe Brand Logo Detection)

本專案使用 HOG 特徵與 SVM 分類器，實現針對 Nike 與 Adidas 商標的自動偵測與辨識。

### 資料集安裝說明 (Dataset Setup)

重要注意：由於 GitHub 單一檔案大小限制，原始圖像資料集已分割為以下三個壓縮檔 (.rar)，請依照以下步驟進行還原，否則程式將無法讀取圖片：

1. 請下載專案中的以下三個壓縮檔：
   - `nike.rar`
   - `adidas.rar`
   - `new_input.rar` (測試用圖片)

2. 將這三個檔案解壓縮。

3. 請在專案根目錄下確認有一個名為 **`dataset`** 的資料夾，並將解壓縮後的資料夾放入其中。

**最終的檔案結構應如下所示：**

```text
Shoe-Brand-Detection/  (專案根目錄)
│
├── conf/              (設定檔資料夾)
├── output/            (模型存放資料夾)
├── ...
│
└── dataset/           <--- 請確保所有圖片都在這裡面
    ├── nike/          <-- 解壓自 nike.rar
    ├── adidas/        <-- 解壓自 adidas.rar
    └── new_input/     <-- 解壓自 new_input.rar
