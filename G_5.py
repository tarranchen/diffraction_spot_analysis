import os
import time
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, amplitude, mean, stddev):
    """定義高斯函數模型"""
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)


def fit_gaussian_peaks(data, max_peaks=10, min_peak_height_ratio=0.05):
    """
    對單列數據進行迭代高斯擬合。

    返回:
    - centers, amplitudes, fwhms: 擬合出的峰值參數
    - processing_time: 處理時間
    - total_fit_curve: 完整的擬合曲線 (背景 + 所有峰)
    """
    start_time = time.time()

    data = np.array(data, dtype=float)
    x_data = np.arange(len(data))

    # 1. 減去線性背景
    background_fit_points = int(len(data) * 0.05)
    if background_fit_points < 2: background_fit_points = 2

    x_bg = np.concatenate([x_data[:background_fit_points], x_data[-background_fit_points:]])
    y_bg = np.concatenate([data[:background_fit_points], data[-background_fit_points:]])

    if len(x_bg) >= 2:
        coeffs = np.polyfit(x_bg, y_bg, 1)
        background = np.polyval(coeffs, x_data)
    else:
        background = np.full_like(data, np.mean(data))

    data_corrected = data - background
    residual_data = np.copy(data_corrected)

    # 初始化一個陣列來儲存所有擬合峰的總和
    all_peaks_sum = np.zeros_like(data, dtype=float)

    peak_threshold = (np.max(data_corrected) - np.min(data_corrected)) * min_peak_height_ratio
    if peak_threshold <= 0:
        total_fit_curve = background
        return [], [], [], time.time() - start_time, total_fit_curve

    centers, amplitudes, fwhms = [], [], []

    # 2 & 3. 迭代尋找高斯峰
    for _ in range(max_peaks):
        current_max_val = np.max(residual_data)
        if current_max_val < peak_threshold: break

        peak_pos = np.argmax(residual_data)
        initial_amplitude, initial_mean = current_max_val, peak_pos

        try:
            half_max = initial_amplitude / 2
            left_idx = np.where(residual_data[:peak_pos] < half_max)[0][-1]
            right_idx = np.where(residual_data[peak_pos:] < half_max)[0][0] + peak_pos
            initial_fwhm_guess = right_idx - left_idx
            initial_stddev = initial_fwhm_guess / (2 * np.sqrt(2 * np.log(2)))
        except (IndexError, ValueError):
            initial_stddev = 10.0

        p0 = [initial_amplitude, initial_mean, initial_stddev]
        if p0[2] <= 0: p0[2] = 1.0

        fit_window = int(initial_fwhm_guess * 2) if 'initial_fwhm_guess' in locals() and initial_fwhm_guess > 0 else 30
        start, end = max(0, peak_pos - fit_window), min(len(residual_data), peak_pos + fit_window)
        x_fit, y_fit = x_data[start:end], residual_data[start:end]

        try:
            params, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0, maxfev=10000)
            amp, mean, std = params
            if std > 0 and mean > 0 and mean < len(data):
                centers.append(mean)
                amplitudes.append(amp)
                fwhms.append(2 * np.sqrt(2 * np.log(2)) * std)

                fitted_peak = gaussian(x_data, amp, mean, std)
                all_peaks_sum += fitted_peak  # 將擬合出的峰加到總和中
                residual_data -= fitted_peak  # 從殘差數據中減去
            else:
                residual_data[start:end] = np.min(residual_data)
        except (RuntimeError, ValueError):
            residual_data[start:end] = np.min(residual_data)
            print(f"警告: 在位置 {peak_pos} 附近的一次擬合未能收斂。")
            continue

    # 最終的擬合曲線 = 背景 + 所有峰的總和
    total_fit_curve = background + all_peaks_sum

    end_time = time.time()
    processing_time = end_time - start_time
    return centers, amplitudes, fwhms, processing_time, total_fit_curve


def main():
    """主執行函數"""
    input_filename = "y_range_sum_result_avg.csv"
    results_filename = "Gu_results.csv"
    fitting_filename = "y_range_sum_result_avg_fitting.csv"  # 新的輸出檔名

    if not os.path.exists(input_filename):
        print(f"錯誤: 輸入檔案 '{input_filename}' 不存在於目前資料夾中。")
        return

    df = pd.read_csv(input_filename, header=0)
    data_df = df.iloc[:, 1:]

    # 初始化兩個列表來儲存兩種不同的輸出結果
    results_list = []
    fitting_data_list = []  # 用於儲存擬合曲線數據

    max_peaks = 10

    print("開始處理數據 (以『列』為單位)...")

    for index, row in data_df.iterrows():
        print(f"正在處理第 {index + 1} 列數據...")
        data_series = row.fillna(0).values

        # 執行擬合，現在會接收 5 個返回值
        centers, amplitudes, fwhms, proc_time, fitted_curve = fit_gaussian_peaks(data_series, max_peaks=max_peaks)

        # --- 準備第一份輸出檔案 (Gu_results.csv) 的數據 ---
        row_result = {}
        spectrum_id = df.iloc[index, 0]
        row_result['Spectrum_ID'] = spectrum_id
        for i in range(max_peaks):
            row_result[f'Position_{i + 1}'] = centers[i] if i < len(centers) else ''
        for i in range(max_peaks):
            row_result[f'Intensity_{i + 1}'] = amplitudes[i] if i < len(amplitudes) else ''
        for i in range(max_peaks):
            row_result[f'FWHM_{i + 1}'] = fwhms[i] if i < len(fwhms) else ''
        row_result['Processing_Time'] = proc_time
        results_list.append(row_result)

        # --- 準備第二份輸出檔案 (y_range_sum_result_avg_fitting.csv) 的數據 ---
        # 將光譜ID與擬合曲線數據結合
        fitting_row = [spectrum_id] + list(fitted_curve)
        fitting_data_list.append(fitting_row)

    # --- 寫入第一份檔案：Gu_results.csv ---
    results_df = pd.DataFrame(results_list)
    pos_cols = [f'Position_{i + 1}' for i in range(max_peaks)]
    amp_cols = [f'Intensity_{i + 1}' for i in range(max_peaks)]
    fwhm_cols = [f'FWHM_{i + 1}' for i in range(max_peaks)]
    final_col_order = ['Spectrum_ID'] + pos_cols + amp_cols + fwhm_cols + ['Processing_Time']
    results_df = results_df[final_col_order]
    results_df.to_csv(results_filename, index=False, float_format='%.4f')
    print(f"\n高斯擬合參數已儲存至 '{results_filename}'。")

    # --- 寫入第二份檔案：y_range_sum_result_avg_fitting.csv ---
    # 使用原始檔案的欄位名稱作為新檔案的標頭
    fitting_df = pd.DataFrame(fitting_data_list, columns=df.columns)
    fitting_df.to_csv(fitting_filename, index=False, float_format='%.4f')
    print(f"完整擬合曲線數據已儲存至 '{fitting_filename}'。")

    print("\n處理完成！")


if __name__ == "__main__":
    main()