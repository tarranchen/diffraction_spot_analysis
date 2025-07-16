import pandas as pd
import numpy as np
import tifffile
import os

def convert_csv_to_tiff(csv_filename="y_range_sum_result.csv", tiff_filename="output.tif"):
    """
    Reads a CSV file with headers in the first row and column,
    and converts its numerical data into a 32-bit TIFF image.

    Args:
        csv_filename (str): The name of the input CSV file.
        tiff_filename (str): The name of the output TIFF file.
    """
    # 建立 CSV 檔案的完整路徑
    # __file__ 代表目前 script 的路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)

    # 檢查 CSV 檔案是否存在
    if not os.path.exists(csv_path):
        print(f"錯誤：找不到檔案 '{csv_path}'。")
        print("請確認 CSV 檔案與此 script 位於同一個資料夾中。")
        return

    try:
        # --- 步驟 1: 讀取 CSV 檔案 ---
        # index_col=0 表示將第一欄作為 DataFrame 的索引
        print(f"正在讀取檔案: {csv_filename}...")
        data_df = pd.read_csv(csv_path, index_col=0)

        # --- 步驟 2: 轉換為 NumPy 陣列 ---
        # 將 DataFrame 轉換為 NumPy 陣列，並指定資料類型為 32-bit float
        # 影像的強度值通常使用浮點數表示
        image_data = data_df.to_numpy().astype(np.float32)
        print(f"成功讀取資料，影像維度為: {image_data.shape[0]} x {image_data.shape[1]}")

        # --- 步驟 3: 將陣列儲存為 32-bit TIFF 檔案 ---
        output_path = os.path.join(script_dir, tiff_filename)
        print(f"正在儲存 32-bit TIFF 檔案至: {output_path}...")
        tifffile.imwrite(output_path, image_data)

        print("\n轉換完成！🎉")

    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")

if __name__ == "__main__":
    convert_csv_to_tiff()