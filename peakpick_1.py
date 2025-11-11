import numpy as np
import tifffile
import csv
# --- *** 修改 1/4 : 匯入 find_objects *** ---
from scipy.ndimage import maximum_filter, minimum_filter, label, center_of_mass, find_objects
import os
import argparse
import sys


def find_peaks(input_file, output_image_file, output_csv_file, xd, yd, h):
    """
    在 32-bit TIFF 影像中尋找 peak。
    (參數說明... )
    """

    # --- *** (日誌部分已加入 flush=True) *** ---
    print(f"--- 開始處理 ---", flush=True)
    print(f"正在讀取影像: {input_file}...", flush=True)
    try:
        image_data = tifffile.imread(input_file)
        if image_data.ndim != 2:
            print(f"錯誤: 輸入影像不是 2D (維度: {image_data.ndim})。", flush=True)
            print("此腳本僅支援單一 2D 影像。", flush=True)
            return
        if image_data.dtype not in [np.float32, np.uint32, np.int32]:
            print(f"警告: 影像不是 32-bit ({image_data.dtype})，將嘗試轉換為 float32...", flush=True)
            image_data = image_data.astype(np.float32)

    except FileNotFoundError:
        print(f"錯誤: 找不到輸入檔案 {input_file}", flush=True)
        return
    except Exception as e:
        print(f"讀取影像時發生錯誤: {e}", flush=True)
        return

    print(f"影像尺寸: {image_data.shape}", flush=True)
    print(f"使用參數: xd={xd}, yd={yd}, h={h}", flush=True)

    filter_size = (2 * yd + 1, 2 * xd + 1)

    print("正在計算局部最大值...", flush=True)
    local_max = maximum_filter(image_data, size=filter_size, mode='constant', cval=-np.inf)

    print("正在計算局部最小值...", flush=True)
    local_min = minimum_filter(image_data, size=filter_size, mode='constant', cval=np.inf)

    is_local_max = (image_data == local_max)
    is_above_threshold = (image_data - local_min) > h
    potential_peaks_mask = is_local_max & is_above_threshold

    print(f"找到 {np.sum(potential_peaks_mask)} 個 potential peak 像素（包含群集）。", flush=True)

    labels, num_features = label(potential_peaks_mask, structure=np.ones((3, 3)))

    print(f"將 potential peaks 分為 {num_features} 個獨立群集。", flush=True)

    # --- *** 修改 2/4 : 開始優化 *** ---
    print(f"正在定位 {num_features} 個群集的中心...", flush=True)

    peak_coordinates = []  # 儲存最終的 (x, y) 座標
    output_image = np.zeros(image_data.shape, dtype=np.uint8)  # 8-bit 黑色影像

    # 1. (優化) 一次性取得所有群集的邊界框 (Bounding Box)
    # slices 是一個包含 num_features 個元組(tuple)的列表
    # 每個元組是 (slice(y_start, y_stop), slice(x_start, x_stop))
    slices = find_objects(labels, num_features)

    if num_features > 0:
        # 2. (優化) 迭代 slices 列表
        for i in range(num_features):
            feature_index = i + 1
            s = slices[i]  # 取得這個群集的 slice

            # 3. (優化) 只在小範圍 (邊界框) 內建立遮罩
            # 這樣 cluster_mask_small 是一個小陣列 (例如 3x3)，而不是 (3599, 7028)
            cluster_mask_small = (labels[s] == feature_index)

            # 4. (優化) 在小遮罩上計算質心
            # center_y_local, center_x_local 是相對於 "小遮罩" 的座標
            center_y_local, center_x_local = center_of_mass(cluster_mask_small)

            # 5. (優化) 在小遮罩上取得座標
            # cluster_coords_local 也是相對於 "小遮罩" 的 (y, x)
            cluster_coords_local = np.argwhere(cluster_mask_small)  # (y, x) local

            if cluster_coords_local.size == 0:
                continue

                # 6. 計算距離
            distances = np.sum((cluster_coords_local - [center_y_local, center_x_local]) ** 2, axis=1)

            # 7. 找到最近的點
            center_pixel_index = np.argmin(distances)

            # 8. 取得該點的 "小遮罩" 座標
            peak_y_local, peak_x_local = cluster_coords_local[center_pixel_index]

            # 9. (優化) 將 "小遮罩" 座標轉換回 "全域" 座標
            peak_y_global = peak_y_local + s[0].start
            peak_x_global = peak_x_local + s[1].start

            # 10. 存入結果
            peak_coordinates.append((peak_x_global, peak_y_global))
            output_image[peak_y_global, peak_x_global] = 255

            # --- *** 修改 3/4 : 加入進度條 *** ---
            if (i + 1) % 10000 == 0:  # 每處理 10000 個群集回報一次
                print(f"  ...已處理 {i + 1} / {num_features} 個群集", flush=True)

    print(f"處理完成，共找到 {len(peak_coordinates)} 個最終 peaks。", flush=True)
    # --- *** 優化結束 *** ---

    # 4. 輸出 8-bit TIFF 影像
    try:
        print(f"正在儲存 8-bit peak 影像至: {output_image_file}...", flush=True)
        tifffile.imwrite(output_image_file, output_image, photometric='minisblack')
    except Exception as e:
        print(f"儲存影像時發生錯誤: {e}", flush=True)

    # 5. 輸出 CSV 檔案
    try:
        print(f"正在儲存 peak 座標至: {output_csv_file}...", flush=True)
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])  # 寫入表頭
            writer.writerows(peak_coordinates)  # 寫入所有座標
    except Exception as e:
        print(f"儲存 CSV 時發生錯誤: {e}", flush=True)

    print(f"--- 處理完畢 ---", flush=True)


def create_dummy_tiff(filename, shape=(256, 256)):
    """建立一個假的 32-bit TIFF 檔案以供測試"""
    print(f"正在建立測試檔案: {filename}")
    # 建立一個基礎
    data = np.zeros(shape, dtype=np.float32)

    # 加入幾個 "peaks"
    # Peak 1 (單點, 會被 h=50 偵測到)
    data[50, 50] = 500

    # Peak 2 (3x3 群集, 中心點 (101, 101) 應被偵測)
    data[100:103, 100:103] = 800

    # Peak 3 (低對比度, 不會被 h=50 偵測到)
    data[150, 150] = 100
    data[150, 151] = 60  # min = 60, 100-60 = 40 < h

    # Peak 4 (L 型群集, 應找到最接近質心的點)
    data[200, 200] = 600
    data[201, 200] = 600
    data[202, 200] = 600
    data[200, 201] = 600
    data[200, 202] = 600

    # 加入一些噪點
    noise = np.random.rand(*shape) * 10
    data += noise

    try:
        tifffile.imwrite(filename, data)
        print(f"測試檔案 '{filename}' 已建立。", flush=True)
    except Exception as e:
        print(f"建立測試檔案時發生錯誤: {e}", flush=True)


if __name__ == "__main__":

    # --- *** 修改 4/4 : (這部分程式碼不變，確保路徑正確) *** ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    default_input_path = os.path.join(script_dir, "shift.tif")
    default_output_img_path = os.path.join(script_dir, "peaks_output.tif")
    default_output_csv_path = os.path.join(script_dir, "peaks_output.csv")

    # 建立一個 argparse 來處理命令列參數
    parser = argparse.ArgumentParser(
        description="在 32-bit TIFF 影像中尋找 peaks。",
        add_help=False,
        epilog="""
使用範例:
1. 建立測試檔案:
   python peak_finder.py --create_test_file
2. 執行 peak 尋找 (使用預設參數):
   python peak_finder.py test_input_32bit.tif
3. 執行 peak 尋找 (自訂參數):
   python peak_finder.py my_image.tif -o_img peaks.tif -o_csv coords.csv -xd 5 -yd 5 -h 100
"""
        , formatter_class=argparse.RawDescriptionHelpFormatter)

    # <<< *** 修改 2 *** : 手動加入 --help 參數 (因為 -h 已被停用)
    parser.add_argument("--help", action="help", help="顯示此幫助訊息並退出。")

    # --- *** 修改 4/5 : 更新 argparse 的預設值 *** ---
    parser.add_argument("-i", "--input_file", type=str,
                        default=default_input_path,
                        help=f"輸入的 32-bit TIFF 檔案路徑。(預設: {default_input_path})")

    parser.add_argument("-o_img", "--output_image", type=str,
                        default=default_output_img_path,
                        help=f"輸出的 8-bit peak 影像檔案路徑。 (預設: {default_output_img_path})")

    parser.add_argument("-o_csv", "--output_csv", type=str,
                        default=default_output_csv_path,
                        help=f"輸出的 peak 座標 CSV 檔案路徑。 (預設: {default_output_csv_path})")
    parser.add_argument("-xd", type=int, default=5, help="X 方向的比較半徑 (pixel)。 (預設: 3)")
    parser.add_argument("-yd", type=int, default=6, help="Y 方向的比較半徑 (pixel)。 (預設: 3)")
    # 現在這一行不會再報錯了
    parser.add_argument("-h", type=float, default=10000, help="Peak 與周圍最小值的最小高度差。 (預設: 50)")
    parser.add_argument("--create_test_file", action="store_true",
                        help="如果指定，將會建立一個名為 'test_input_32bit.tif' 的測試檔案並退出。")

    args = parser.parse_args()

    if args.create_test_file:
        # 讓測試檔案也建立在腳本資料夾中
        test_file_path = os.path.join(script_dir, "test_input_32bit.tif")
        create_dummy_tiff(test_file_path, shape=(256, 256))
        print(f"請使用 'python peak_finder.py -i {test_file_path}' 來執行分析。", flush=True)
        sys.exit(0)

    if not os.path.exists(args.input_file):
        print(f"錯誤: 找不到輸入檔案 {args.input_file}", flush=True)
        # --- *** 修改 5/5 : 更新提示訊息 *** ---
        if args.input_file == default_input_path:
            print(f"請確保 'shift.tif' 檔案與您的腳本在同一個資料夾中:", flush=True)
            print(f"{script_dir}", flush=True)
        print("您也可以使用 -i <filename> 來指定不同的檔案，或使用 --create_test_file 建立測試檔。", flush=True)
        sys.exit(1)

    find_peaks(args.input_file, args.output_image, args.output_csv, args.xd, args.yd, args.h)