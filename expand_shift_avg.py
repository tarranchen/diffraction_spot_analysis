import csv
import os
import math

def process_data_groups(input_filename="y_range_sum_result.csv"):
    """
    讀取指定的CSV檔案，將資料分組處理，並產生三種輸出檔案：
    1. 一個包含所有處理後資料的合併檔。
    2. 每個資料組的獨立檔案。
    3. 一個包含每組資料之逐行(column-wise)平均值的摘要檔。
    """
    # 檢查輸入檔案是否存在
    if not os.path.exists(input_filename):
        print(f"錯誤: 找不到輸入檔案 '{input_filename}'")
        return

    print(f"正在讀取檔案: {input_filename}...")
    
    # 建立輸出檔案名稱的基礎部分
    base, ext = os.path.splitext(input_filename)
    output_base_name = base

    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            # 讀取所有行，並跳過第一行列標頭
            all_data_rows = list(reader)[1:]

        num_rows_per_group = 100
        # 計算總共有多少組資料
        num_groups = len(all_data_rows) // num_rows_per_group

        if num_groups == 0:
            print("檔案中的資料不足100列，無法形成任何資料組。")
            return

        print(f"檔案包含 {len(all_data_rows)} 列資料，共計 {num_groups} 組。")

        all_processed_rows_for_combined_file = []
        all_average_summary_rows = []

        # 逐一處理每個資料組
        for group_idx in range(num_groups):
            print(f"\n正在處理第 {group_idx + 1} 組資料...")
            
            start_row = group_idx * num_rows_per_group
            end_row = start_row + num_rows_per_group
            current_group_data = all_data_rows[start_row:end_row]

            processed_rows_for_this_group = []

            # --- 第一步 & 第二步：擴增與平移 ---
            # 在組內處理每一列 (共100列)
            for i, row in enumerate(current_group_data):
                if not row:
                    continue

                row_header = row[0]
                original_data = row[1:]
                expanded_data = [item for item in original_data for _ in range(5)]

                shift_amount = 32 * ((num_rows_per_group - 1) - i)
                padding = [''] * shift_amount
                final_row = [row_header] + padding + expanded_data
                processed_rows_for_this_group.append(final_row)

            # --- 第三步：計算每「行」(Column)的平均值 ---
            # 找出這個處理後資料組中最長的列有多少行
            max_cols = 0
            if processed_rows_for_this_group:
                max_cols = max(len(r) for r in processed_rows_for_this_group)

            column_averages = []
            # 從第1行開始計算 (跳過第0行的header)
            for col_idx in range(1, max_cols):
                column_values = []
                for row_data in processed_rows_for_this_group:
                    # 檢查此列是否有足夠的行
                    if col_idx < len(row_data):
                        item = row_data[col_idx]
                        try:
                            # 嘗試將項目轉換為浮點數
                            column_values.append(float(item))
                        except (ValueError, TypeError):
                            # 如果轉換失敗 (例如，儲存格為空或非數字)，則跳過
                            continue
                
                if column_values:
                    # 計算平均值並四捨五入到整數
                    avg = round(sum(column_values) / len(column_values))
                    column_averages.append(avg)
                else:
                    # 如果行中沒有有效數字，則平均值為0
                    column_averages.append(0)
            
            # --- 儲存此資料組的獨立檔案 ---
            group_output_filename = f"{output_base_name}_shift_{group_idx + 1:02d}.csv"
            with open(group_output_filename, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                writer.writerows(processed_rows_for_this_group)
            print(f"已儲存單組資料至: {group_output_filename}")

            # 將處理完的資料加入到合併列表
            all_processed_rows_for_combined_file.extend(processed_rows_for_this_group)
            
            # 建立此組的平均值摘要列
            summary_row = [f"no.{group_idx + 1:02d}"] + column_averages
            all_average_summary_rows.append(summary_row)

        # --- 儲存合併後的總檔案 ---
        combined_output_filename = f"{output_base_name}_shift.csv"
        with open(combined_output_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(all_processed_rows_for_combined_file)
        print(f"\n已儲存所有合併資料至: {combined_output_filename}")

        # --- 儲存平均值摘要檔案 ---
        avg_output_filename = f"{output_base_name}_avg.csv"
        with open(avg_output_filename, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(all_average_summary_rows)
        print(f"已儲存平均值摘要至: {avg_output_filename}")

    except FileNotFoundError:
        # 這是備用，雖然前面已經檢查過一次
        print(f"錯誤: 找不到檔案 {input_filename}")
    except Exception as e:
        print(f"處理檔案時發生未預期的錯誤: {e}")


def main():
    """
    主函數，執行資料處理流程。
    """
    process_data_groups()
    print("\n所有處理流程已完成。")


if __name__ == "__main__":
    main()
