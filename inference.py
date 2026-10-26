import shutil
import time
import xarray as xr
import numpy as np
import pandas as pd
import torch
import sys
import os
import subprocess
import platform

# 添加当前目录到 sys.path
sys.path.append(".")

from datetime import datetime, timedelta
from tqdm import tqdm
from utils.init import load_3h_model, load_6h_model, load_9h_model, load_12h_model, get_vars
from utils.handle import open_asm_merra, stand, anti_stand, convert2dataarray
from utils.scheduler import rolling_model

# 更新 asm 文件
def update_asm(year, month, day, hour, minute, temp_folder):
    url = {
        "GEOS.fp.asm.tavg3_3d_asm_Nv": f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{year}/M{month}/D{day}/GEOS.fp.asm.tavg3_3d_asm_Nv.{year + month + day}_{hour + minute}.V01.nc4",
        "GEOS.fp.asm.tavg1_2d_flx_Nx": f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{year}/M{month}/D{day}/GEOS.fp.asm.tavg1_2d_flx_Nx.{year + month + day}_{hour + minute}.V01.nc4",
        "GEOS.fp.asm.tavg3_2d_aer_Nx": f"https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{year}/M{month}/D{day}/GEOS.fp.asm.tavg3_2d_aer_Nx.{year + month + day}_{hour + minute}.V01.nc4"
    }

    tag_list = []
    for key, value in url.items():
        save_folder = f"{temp_folder}/{key}/"
        update_folder(save_folder)
        tag = download_file(value, save_folder)

        # 如果下载失败，跳过当前文件
        if not tag:
            tag_list.append(False)
            break
        tag_list.append(True)
    return tag_list

# 下载文件到指定文件夹，如果文件已存在且完整，则跳过下载
def download_file(url, download_folder="/data/GEOS-FP", local_filename=None):
    if local_filename is None:
        local_filename = os.path.basename(url)
    file_path = os.path.join(download_folder, local_filename)

    # 替换文件名中的 nc4 为 hdf
    file_path = file_path.replace("nc4", "hdf")
    if os.path.exists(file_path):
        try:
            xr.open_dataset(file_path)
            print(f"文件已存在且完整: {file_path}")
            return True
        except Exception as e:
            print(f"文件已存在，但无法打开，错误: {e}")
            print("将继续下载文件...")
    print(f"开始下载文件: {url}")
    os.makedirs(download_folder, exist_ok=True)
    try:
        system = platform.system().lower()
        if system == "windows":
            wget_command = ["./wget.exe", "-c", url, "-O", file_path]
        elif system == "linux":
            wget_command = ["wget", "-c", url, "-O", file_path]
        subprocess.run(wget_command, check=True)
        # 验证下载的文件是否有效
        try:
            ds = xr.open_dataset(file_path)
            ds.close()
            print(f"文件已成功下载到: {file_path}")
            return True
        except Exception as e:
            print(f"下载的文件无法打开: {file_path}, 错误: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)  # 删除无效文件
            return False
    except subprocess.CalledProcessError as e:
        print(f"下载文件时出现错误: {e}")
        return False

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

# 获取初始字段
def get_init_field(year, month, day, hour, minute):
    init_time = f"{year}-{month}-{day} {hour}:{minute}"

    # 调用 open_asm_merra 获取初始场数据
    result = open_asm_merra(init_time)
    if result is None:
        raise ValueError(f"open_asm_merra 返回 None，输入时间: {init_time}")
    
    # 参数解析：
    # init_field: 原始初始场 NumPy 数组，形状为 (54, latitude, longitude)
    # index: 长度为 54 的变量名列表，与初始场通道一一对应
    # lon: 一维经度数组，长度为经度网格数
    # lat: 一维纬度数组，长度为纬度网格数
    # _: 未使用的文件时间标识字符串（YYYYMMDD_HHMM）
    init_field, index, lon, lat, _ = result

    # 54 个通道 = 12 个气溶胶变量 + 5 个通量及相关变量 + 4 个多层变量 × 9 层 + SLP。
    # 具体变量及层编号见 utils/init.py 的 get_vars()。
    # stand 按通道执行标准化：(数据 - 均值) / 标准差
    init_field = stand(index, init_field)
    # 转为 PyTorch 张量并添加 batch 维度，返回形状为 (1, 54, latitude, longitude)
    init_field = torch.from_numpy(init_field).to(torch.float16).unsqueeze(0).float()
    return  init_field, init_time, index, lon, lat

# 获取数据数组信息
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

    # 定义日期范围
    date_range = pd.date_range(
        start="2025-03-19 19:30:00",
        end="2025-03-19 19:30:00",
        freq="1D"
    )

    # tqdm 是一个进度条库，用于显示处理进度
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
            
            # 检查输出文件夹是否为空，非空则跳过
            if os.path.exists(local_data) and os.listdir(local_data):
                print(f"输出文件夹 {local_data} 非空，跳过处理")
                processed_times.add(time_key)
                continue
                
            # 更新文件夹（如果非空会返回False），非空则跳过
            if not update_folder(local_data):
                processed_times.add(time_key)
                continue

            # 下载初始场数据，返回一个布尔列表，每个元素对应一个变量是否下载成功
            tag = update_asm(year, month, day, hour, minute, temp_asm_folder)
            if False in tag:
                print(f"数据下载失败，跳过 {time_key}...")
                continue

            # 参数分析：
            # init_field: 标准化后的初始场张量，形状为 (1, 54, latitude, longitude)
            # init_time: 起报时间字符串（YYYY-MM-DD HH:MM）
            # index: 长度为 54 的变量名列表，与通道一一对应
            # lon: 一维经度数组
            # lat: 一维纬度数组
            init_field, init_time, index, lon, lat = get_init_field(year, month, day, hour, minute)
            time_range = pd.date_range(start=pd.Timestamp(init_time),
                                       end=pd.Timestamp(init_time) + pd.Timedelta(hours=40 * 3), freq="3h")
                                       
            info = get_dataarray_info()

            # 为当前起报时间加载模型，供后续 40 个目标时效的预报复用
            model_3h = load_3h_model(device=device)
            model_6h = load_6h_model(device=device)
            model_9h = load_9h_model(device=device)
            model_12h = load_12h_model(device=device)

            # 这里为什么是 40 个预报步长？
            # 因为每个预报步长都是 3 小时，所以 40 个预报步长就是 120 小时，符合我们的需求
            for i in tqdm(range(40), desc="预报步数"):
                target_step = i + 1
                
                # rolling_model 是一个函数，用于滚动预报
                # 它接受初始场数据、目标步长、索引、模型实例、设备、方法和开始时间
                # 返回目标时刻反标准化后的 NumPy 数组，形状为 (54, latitude, longitude)
                # index 是与数据通道一一对应的变量名列表
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

