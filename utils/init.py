import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy
import torch
import os
import argparse
import xarray as xr
import argparse
import sys
import emoji
import re
import shutil
import yaml
# import weatherbench2

from matplotlib import pyplot as plt

sys.path.append(".")

from tqdm import tqdm


# 规定MERRA-2变量
def get_vars():

    # vars_dict = {
    #     "tavg1_2d_aer_Nx": ['BCEXTTAU', 'BCSMASS', 'DUEXTTAU', 'DUSMASS', 'OCEXTTAU',
    #                         'OCSMASS', 'SO4SMASS', 'SSEXTTAU', 'SSSMASS', 'SUEXTTAU',
    #                         'TOTEXTTAU', 'TOTSCATAU'],
    #     "tavg1_2d_flx_Nx": ['QLML', 'TLML', 'ULML', 'VLML', 'PBLH', 'PRECTOT'],
    #     "tavg3_3d_asm_Nv": {"var": ["QV", "SLP", "T", "U", "V"],
    #                         "level": [45, 48, 51, 53, 56, 60, 63, 68, 72]}
    # }

    # # V0
    # vars_dict = {
    #     "tavg1_2d_aer_Nx": ['DUEXTTAU', 'DUSMASS', 'TOTEXTTAU'],
    #     "tavg1_2d_flx_Nx": ['QLML', 'TLML', 'ULML', 'VLML', 'PBLH', 'PRECTOT'],
    #     "tavg3_3d_asm_Nv": {"var": ["QV", "SLP", "T", "U", "V"],
    #                         "level": [45, 48, 51, 53, 56, 60, 63, 68, 72]}
    # }

    # V1：每个变量的二维空间场占一个通道；多层变量的每个选定层分别占一个通道。
    # 总通道数 = 12 个气溶胶变量 + 5 个通量及相关变量
    #          + 4 个多层变量（QV、T、U、V）× 9 层 + 1 个二维变量（SLP）= 54。
    vars_dict = {
        # 气溶胶二维场，共 12 个通道。
        # 此处使用 tavg1 配置键；open_asm_merra 读取时将其映射到 tavg3 目录。
        "tavg1_2d_aer_Nx": [
            'BCEXTTAU',   # 黑碳气溶胶消光光学厚度
            'BCSMASS',    # 黑碳气溶胶地表质量浓度
            'DUEXTTAU',   # 沙尘气溶胶消光光学厚度
            'DUSMASS',    # 沙尘气溶胶地表质量浓度
            'OCEXTTAU',   # 有机碳气溶胶消光光学厚度
            'OCSMASS',    # 有机碳气溶胶地表质量浓度
            'SO4SMASS',   # 硫酸盐气溶胶地表质量浓度
            'SSEXTTAU',   # 海盐气溶胶消光光学厚度
            'SSSMASS',    # 海盐气溶胶地表质量浓度
            'SUEXTTAU',   # 硫酸盐气溶胶消光光学厚度
            'TOTEXTTAU',  # 总气溶胶消光光学厚度
            'TOTSCATAU',  # 总气溶胶散射光学厚度
        ],
        # 地表通量及相关二维场，共 5 个通道。
        "tavg1_2d_flx_Nx": [
            'QLML',     # 最低模型层比湿
            'TLML',     # 最低模型层温度
            'ULML',     # 最低模型层纬向风分量
            'VLML',     # 最低模型层经向风分量
            'PRECTOT',  # 总降水率
        ],
        # 同化大气状态场，共 4 × 9 + 1 = 37 个通道。
        "tavg3_3d_asm_Nv": {
            "var": [
                "QV",   # 比湿：取下列 9 层，形成 9 个通道
                "SLP",  # 海平面气压：二维场，仅 1 个通道，不按层展开
                "T",    # 温度：取下列 9 层，形成 9 个通道
                "U",    # 纬向风分量：取下列 9 层，形成 9 个通道
                "V",    # 经向风分量：取下列 9 层，形成 9 个通道
            ],
            # 模型垂直层编号，不是气压值；每个多层变量均读取这 9 层。
            "level": [45, 48, 51, 53, 56, 60, 63, 68, 72],
        },
    }

    return vars_dict


# 初始化三小时模型（改为加载 JIT traced 模型）
def load_3h_model(device):
    print(emoji.emojize(':black_circle: loading 3h model'))
    # 修改为加载 JIT traced 模型
    model_3h = torch.jit.load("./model/gamfs_3h_traced.pt", map_location=device)
    model_3h = model_3h.to(device)
    model_3h.eval()  

    return model_3h

# 初始化6小时模型（改为加载 JIT traced 模型）
def load_6h_model(device):
    print(emoji.emojize(':black_circle: loading 6h model'))
    # 修改为加载 JIT traced 模型
    model_6h = torch.jit.load("./model/gamfs_6h_traced.pt", map_location=device)
    model_6h = model_6h.to(device)
    model_6h.eval()  

    return model_6h

# 初始化9小时模型（改为加载 JIT traced 模型）
def load_9h_model(device):
    """
    导入9小时预报模型
    @param device: 导入设备
    @return: eval状态下的模型
    """
    print(emoji.emojize(':black_circle: loading 9h model'))
    # 修改为加载 JIT traced 模型
    model_9h = torch.jit.load("./model/gamfs_9h_traced.pt", map_location=device)
    model_9h = model_9h.to(device)
    model_9h.eval()  

    return model_9h

# 初始化12小时模型（改为加载 JIT traced 模型）
def load_12h_model(device):
    """
    导入12小时预报模型
    @param device: 导入设备
    @return: eval状态下的模型
    """
    print(emoji.emojize(':black_circle: loading 12h model'))
    # 修改为加载 JIT traced 模型
    model_12h = torch.jit.load("./model/gamfs_12h_traced.pt", map_location=device)
    model_12h = model_12h.to(device)
    model_12h.eval() 

    return model_12h


if __name__ == "__main__":

    model = load_12h_model("TransUnet", "cuda:0")
    torch.save(model, "/data/_project_zxt/merra_forecast/checkpoint/gamfs_models/gamfs_12h.pth")




