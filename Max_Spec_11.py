import os
import csv
import numpy as np
from PIL import Image

def get_neighbor_average(image_array, bad_pixel_coord, all_bad_pixels_set):
    """
    計算指定壞點周圍8個鄰居的平均像素值。

    在計算平均時，會忽略其他也是壞點的鄰居。

    Args:
        image_array (np.ndarray): 原始的、未經修改的影像陣列。
        bad_pixel_coord (tuple): 要計算的壞點座標 (x, y)。
        all_bad_pixels_set (set): 包含所有壞點座標的集合。

    Returns:
        int: 周圍有效鄰居像素值的平均值（四捨五入到整數）。如果沒有有效鄰居，則回傳0。
    """
    x, y = bad_pixel_coord
    neighbor_values = []
    img_height, img_width = image_array.shape

    # 遍歷 3x3 的鄰居網格
    for j in range(y - 1, y + 2):
        for i in range(x - 1, x + 2):
            # 跳過中心點（壞點本身）
            if i == x and j == y:
                continue
            
            # 檢查鄰居座標是否在影像範圍內
            if 0 <= j < img_height and 0 <= i < img_width:
                # 檢查鄰居是否不是壞點
                if (i, j) not in all_bad_pixels_set:
                    neighbor_values.append(image_array[j, i])

    if not neighbor_values:
        return 0
    
    # 計算平均值並四捨五入到整數
    return int(round(np.mean(neighbor_values)))

def calculate_centroid(matrix):
    """
    計算給定 2D 矩陣的質心（權重中心）。
    座標索引從 0 開始。

    Args:
        matrix (np.ndarray): 輸入的二維 NumPy 陣列 (例如：影像的一個區域)。

    Returns:
        tuple: 包含 (質心X, 質心Y) 的元組。如果權重為0，則回傳 (0, 0)。
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("輸入必須是一個二維的 NumPy 陣列")

    total_weight = np.sum(matrix)
    if total_weight == 0:
        return (0, 0)

    num_rows, num_cols = matrix.shape
    
    # 創建對應於 matrix 內部索引的座標網格 (0 到 n-1)
    xx, yy = np.meshgrid(np.arange(num_cols), np.arange(num_rows))

    # 計算加權和 (Moments)
    moment_x = np.sum(xx * matrix)
    moment_y = np.sum(yy * matrix)

    # 計算質心在 matrix 內的局部座標
    centroid_x_local = moment_x / total_weight
    centroid_y_local = moment_y / total_weight

    return (centroid_x_local, centroid_y_local)

def load_bad_pixels(folder_path, csv_filename="bad_pixels.csv"):
    """
    從指定的 CSV 檔案中讀取壞點座標。
    """
    bad_pixels_set = set()
    csv_path = os.path.join(folder_path, csv_filename)

    if not os.path.exists(csv_path):
        print(f"[資訊] 找不到壞點資料庫 '{csv_filename}'。將不忽略任何像素。")
        return bad_pixels_set

    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if not row or len(row) < 2:
                    continue
                try:
                    x = int(row[0].strip())
                    y = int(row[1].strip())
                    bad_pixels_set.add((x, y))
                except ValueError:
                    print(f"[警告] 無法解析 '{csv_filename}' 第 {i+1} 行的座標: {row}。已忽略此行。")
        
        if bad_pixels_set:
            print(f"[*] 已成功從 '{csv_filename}' 載入 {len(bad_pixels_set)} 個壞點座標。")
        else:
            print(f"[警告] 壞點資料庫 '{csv_filename}' 為空或格式不正確。")

    except Exception as e:
        print(f"[錯誤] 無法讀取壞點資料庫 '{csv_filename}'。原因: {e}")

    return bad_pixels_set


def find_max_pixel_positions(folder_path):
    """
    分析資料夾中所有 TIF 影像，找出最大值位置，並進行三次迭代計算質心，最後輸出簡化結果。
    同時，計算指定Y軸範圍內各X行的像素總和，並輸出到另一個CSV檔案。
    """
    # ==================================================================
    # === 設定區: 請在此處修改您需要的參數 ===
    # ==================================================================
    # --- 質心計算設定 ---
    # 設定區域的總寬度 (必須是奇數)
    REGION_WIDTH = 21
    # 設定區域的總高度 (必須是奇數)
    REGION_HEIGHT = 29

    # --- Y軸範圍加總設定 ---
    # 設定Y軸加總的起始行 (包含此行)
    Y_SUM_START_ROW = 177
    # 設定Y軸加總的結束行 (包含此行)
    Y_SUM_END_ROW = 220
    # ==================================================================

    # 自動計算向外延伸的像素數 (例如，寬度11 -> 左右各延伸5)
    half_width = (REGION_WIDTH - 1) // 2
    half_height = (REGION_HEIGHT - 1) // 2

    print("--- 影像分析腳本 v7.1 (壞點平均化) ---")
    print(f"[*] 質心分析區域大小: 寬度 {REGION_WIDTH} x 高度 {REGION_HEIGHT}")
    print(f"[*] Y軸加總範圍: 從第 {Y_SUM_START_ROW} 行到第 {Y_SUM_END_ROW} 行")
    bad_pixels = load_bad_pixels(folder_path)

    # 準備存放兩份不同報告的資料
    centroid_output_data = []
    y_sum_output_data = []
    error_files = []
    
    # 用於建立Y軸加總報告的標頭
    y_sum_header_generated = False
    y_sum_header = ['Filename']

    print(f"\n[*] 開始掃描影像資料夾：{folder_path}")

    file_list = sorted(os.listdir(folder_path))

    for filename in file_list:
        if filename.lower().endswith(('.tif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
                img_height, img_width = img_array.shape

                if img_array.ndim != 2:
                    raise ValueError(f"非單通道灰階影像，影像模式(mode)={img.mode}")
                if img_array.size == 0:
                    raise ValueError("影像為空或無法讀取")

                # *** 壞點處理邏輯更新 ***
                # 建立一個可修改的複本，用於後續分析
                search_array = img_array.copy()
                if bad_pixels:
                    for x_bad, y_bad in bad_pixels:
                        if 0 <= y_bad < img_height and 0 <= x_bad < img_width:
                            # 計算周圍像素的平均值
                            avg_val = get_neighbor_average(img_array, (x_bad, y_bad), bad_pixels)
                            # 在複本中用平均值取代壞點
                            search_array[y_bad, x_bad] = avg_val

                # --- 任務1: 質心計算 ---
                max_val_search = np.max(search_array)
                max_pos = np.argwhere(search_array == max_val_search)[0]
                y, x = max_pos
                original_max_val = img_array[y, x]
                print(f"    - 已處理: {filename}，最大值位置 (X,Y): ({x}, {y})")

                y_start_1 = max(0, y - half_height)
                y_end_1 = min(img_height, y + half_height + 1)
                x_start_1 = max(0, x - half_width)
                x_end_1 = min(img_width, x + half_width + 1)
                roi_1 = img_array[y_start_1:y_end_1, x_start_1:x_end_1]
                centroid_x_local_1, centroid_y_local_1 = calculate_centroid(roi_1)
                centroid_x_global_1 = x_start_1 + centroid_x_local_1
                centroid_y_global_1 = y_start_1 + centroid_y_local_1
                
                center_x_round_2 = int(round(centroid_x_global_1))
                center_y_round_2 = int(round(centroid_y_global_1))
                y_start_2 = max(0, center_y_round_2 - half_height)
                y_end_2 = min(img_height, center_y_round_2 + half_height + 1)
                x_start_2 = max(0, center_x_round_2 - half_width)
                x_end_2 = min(img_width, center_x_round_2 + half_width + 1)
                roi_2 = img_array[y_start_2:y_end_2, x_start_2:x_end_2]
                centroid_x_local_2, centroid_y_local_2 = calculate_centroid(roi_2)
                centroid_x_global_2 = x_start_2 + centroid_x_local_2
                centroid_y_global_2 = y_start_2 + centroid_y_local_2

                center_x_round_3 = int(round(centroid_x_global_2))
                center_y_round_3 = int(round(centroid_y_global_2))
                y_start_3 = max(0, center_y_round_3 - half_height)
                y_end_3 = min(img_height, center_y_round_3 + half_height + 1)
                x_start_3 = max(0, center_x_round_3 - half_width)
                x_end_3 = min(img_width, center_x_round_3 + half_width + 1)
                roi_3 = img_array[y_start_3:y_end_3, x_start_3:x_end_3]
                
                final_region_sum = np.sum(roi_3)
                centroid_x_local_3, centroid_y_local_3 = calculate_centroid(roi_3)
                final_centroid_x = x_start_3 + centroid_x_local_3
                final_centroid_y = y_start_3 + centroid_y_local_3
                
                print(f"      -> 最終質心 (X,Y): ({final_centroid_x:.2f}, {final_centroid_y:.2f})")
                
                centroid_output_data.append([
                    filename, int(x), int(y), int(original_max_val), 
                    f"{final_centroid_x:.4f}", f"{final_centroid_y:.4f}", int(final_region_sum)
                ])

                # --- 任務2: Y軸範圍加總 ---
                y_slice = search_array[Y_SUM_START_ROW : Y_SUM_END_ROW + 1, :]
                column_sums = np.sum(y_slice, axis=0)
                y_sum_output_data.append([filename] + column_sums.tolist())

                # 僅在處理第一張圖片後，動態生成標頭
                if not y_sum_header_generated:
                    y_sum_header.extend([f'Sum_Col_{i}' for i in range(img_width)])
                    y_sum_header_generated = True

            except Exception as e:
                print(f"[錯誤] 無法處理檔案：{filename}，原因：{e}")
                error_files.append(filename)

    # --- 寫入第一份報告: 質心結果 ---
    if centroid_output_data:
        output_csv_1 = os.path.join(folder_path, 'max_pixel_positions_result.csv')
        try:
            with open(output_csv_1, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Filename', 'Max_X', 'Max_Y', 'Max_Value', 'Centroid_X', 'Centroid_Y', 'Region_Sum'])
                writer.writerows(centroid_output_data)
            print(f"\n✅ 質心分析報告已儲存到：{output_csv_1}")
        except IOError as e:
            print(f"\n[重大錯誤] 無法寫入質心報告！原因：{e}")
    else:
        print("\n[注意] 沒有產生任何質心分析資料。")

    # --- 寫入第二份報告: Y軸加總結果 ---
    if y_sum_output_data:
        output_csv_2 = os.path.join(folder_path, 'y_range_sum_result.csv')
        try:
            with open(output_csv_2, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(y_sum_header)
                writer.writerows(y_sum_output_data)
            print(f"✅ Y軸範圍加總報告已儲存到：{output_csv_2}")
        except IOError as e:
            print(f"\n[重大錯誤] 無法寫入Y軸加總報告！原因：{e}")
    else:
        print("\n[注意] 沒有產生任何Y軸加總資料。")


    if error_files:
        print(f"\n⚠ 有 {len(error_files)} 個檔案處理失敗，清單如下：")
        for f in error_files:
            print(f"  - {f}")

# --- 主程式執行區塊 ---
if __name__ == "__main__":
    current_folder_path = os.getcwd()
    find_max_pixel_positions(current_folder_path)
    # input("\n請按 Enter 鍵結束程式...")
