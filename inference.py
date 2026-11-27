import shutil
import numpy as np
import pandas as pd
import torch
import sys
import os
import subprocess
from pathlib import Path

sys.path.append(".")

from datetime import datetime
from tqdm import tqdm
from download_data import fetch
from utils.datasets import PRODUCTS
from utils.init import load_3h_model, load_6h_model, load_9h_model, load_12h_model
from utils.handle import open_asm_merra, stand, anti_stand, convert2dataarray
from utils.scheduler import rolling_model

def update_asm(year, month, day, hour, minute, temp_folder):
    stamp = datetime.strptime(year + month + day + hour + minute, "%Y%m%d%H%M")
    results = []
    for product in PRODUCTS:
        if product.merra_host:
            continue
        try:
            fetch(Path(temp_folder), product, stamp)
        except (OSError, ValueError, AttributeError, TypeError, subprocess.CalledProcessError) as error:
            print(f"GEOS-FP {product.name} {stamp}: {error}")
            results.append(False)
            break
        results.append(True)
    return results


def update_folder(folder):
    """检查文件夹是否存在且是否为空"""
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True  # 新建文件夹，可以继续任务
    else:
        # 检查文件夹是否为空
        if not os.listdir(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)
            return True  # 空文件夹，已重新创建，可以继续任务
        else:
            print(f"文件夹 {folder} 非空，跳过处理")
            return False  # 非空文件夹，跳过任务

def get_init_field(year, month, day, hour, minute):
    init_time = f"{year}-{month}-{day} {hour}:{minute}"
    result = open_asm_merra(init_time)
    if result is None:
        raise ValueError(f"open_asm_merra 返回 None，输入时间: {init_time}")
    init_field, index, lon, lat, _ = result
    init_field = stand(index, init_field)
    init_field = torch.from_numpy(init_field).to(torch.float16).unsqueeze(0).float()
    return init_field, init_time, index, lon, lat

def get_dataarray_info():
    with open('./info.pkl', 'rb') as f:
        import pickle
        loaded_info = pickle.load(f)
    return loaded_info

if __name__ == "__main__":

    device = "cuda:0"
    temp_asm_folder = "./temp_asm/"
    output_folder = "./inference/"
    variables = ['TLML', 'ULML', 'VLML', 'SLP', 'BCEXTTAU', 'BCSMASS', 'DUEXTTAU', 'DUSMASS', 'OCEXTTAU', 'OCSMASS', 'SO4SMASS', 'SSEXTTAU', 'SSSMASS', 'SUEXTTAU', 'TOTEXTTAU', 'TOTSCATAU']

    processed_times = set()

    date_range = pd.date_range(
        start="2024-07-26 22:30:00",
        end="2024-07-26 22:30:00",
        freq="1D"
    )

    for current_time in tqdm(date_range, desc="处理日期"):
        try:
            year = current_time.strftime("%Y")
            month = current_time.strftime("%m")
            day = current_time.strftime("%d")
            hour = current_time.strftime("%H")
            minute = current_time.strftime("%M")
            time_key = f"{year}{month}{day}_{hour}{minute}"

            if time_key in processed_times:
                print(f"时间 {time_key} 已处理，跳过...")
                continue

            print(f"处理时间：{year}-{month}-{day} {hour}:{minute}")
            local_data = f"{output_folder}/{time_key}/"
            
            # 检查输出文件夹是否为空
            if os.path.exists(local_data) and os.listdir(local_data):
                print(f"输出文件夹 {local_data} 非空，跳过处理")
                processed_times.add(time_key)
                continue
                
            # 更新文件夹（如果非空会返回False）
            if not update_folder(local_data):
                processed_times.add(time_key)
                continue

            tag = update_asm(year, month, day, hour, minute, temp_asm_folder)
            if False in tag:
                print(f"数据下载失败，跳过 {time_key}...")
                continue

            init_field, init_time, index, lon, lat = get_init_field(year, month, day, hour, minute)
            time_range = pd.date_range(start=pd.Timestamp(init_time),
                                       end=pd.Timestamp(init_time) + pd.Timedelta(hours=40 * 3), freq="3h")

            info = get_dataarray_info()

            model_3h = load_3h_model(device=device)
            model_6h = load_6h_model(device=device)
            model_9h = load_9h_model(device=device)
            model_12h = load_12h_model(device=device)

            for i in tqdm(range(40), desc="预报步数"):
                target_step = i + 1
                output = rolling_model(init_field.clone(),
                                       target_step,
                                       index,
                                       model_3h=model_3h,
                                       model_6h=model_6h,
                                       model_9h=model_9h,
                                       model_12h=model_12h,
                                       device=device,
                                       method="3+6+9+12",
                                       start_time=pd.Timestamp(init_time))

                data_array, _ = convert2dataarray(
                    info,
                    output[np.newaxis, :, :, :],
                    index,
                    pd.DatetimeIndex([time_range[i + 1]]),
                    lat,
                    lon
                )
                save_path = f"{local_data}/AI_GAMFS.{time_range[0].strftime('%Y%m%d_%H%M')}+{time_range[i + 1].strftime('%Y%m%d_%H%M')}.V01.nc"
                data_array = data_array[variables]
                data_array.to_netcdf(save_path)

            del model_3h, model_6h, model_9h, model_12h
            del init_field, output, data_array
            torch.cuda.empty_cache()

            print(f"数据已保存至 {local_data}")
            processed_times.add(time_key)
            print(f"时间 {time_key} 已记录为已处理")

        except Exception as e:
            print(f"处理 {time_key} 时发生错误: {str(e)}")
            continue

