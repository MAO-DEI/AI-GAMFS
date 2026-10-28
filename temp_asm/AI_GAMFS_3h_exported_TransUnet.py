# AI-GAMFS 3h 导出模型：中文阅读注释版
# ================================================================
# 这是什么：从 gamfs_3h_traced.pt 提取出的 TorchScript 计算表示。
# 这不是什么：作者的完整 Python 源码、训练脚本或可独立运行的模型定义。
# Module、Tensor、__torch__、ops、CONSTANTS 等由 TorchScript 环境提供；
# 本文件保留导出运算，只添加注释。不要直接运行或将它当作训练模型导入。
#
# 推荐阅读顺序：
# 1. TransUnet：先看整体输入输出和模块连接。
# 2. ViT：看空间特征怎样变成 token（一个 token 是一个特征向量）。
# 3. MultiHeadSelfAttention：看 token 怎样相互交换信息。
# 4. TransformerBlock / TransformerEncoder：看注意力怎样组成完整网络。
# 5. Bottleneck / Up：补充卷积编码和空间解码细节。
#
# 维度记号：B=批量大小，C=通道数，H/W=空间高/宽，N=空间 token 数，
# D=token 的特征维数，L=N+1（加入 cls_token 后），h=注意力头数，d=每头维数。
# “通道”存不同变量或特征；“token”是 Transformer 操作的序列单位。
#
# 导出语法速查：
# - _0、_28、argument_1 等是自动生成的局部名字，不是有物理意义的变量名。
#   不同 forward 内同名临时变量互不相关。
# - self.xxx 是子模块/参数；xxx = self.xxx 只是取引用，不会重新建模或训练。
# - __parameters__/__buffers__ 是当前模块直接持有的参数/缓冲区清单。
#   顶层列表为空，不代表子模块没有可学习参数。
# - ___torch_mangle_* 是导出时用于区分类/实例的名字，不表示一种新算法。
# - ops.prim.NumToTensor(torch.size(x, k)) 再 int(...)：读取第 k 维长度。
# - torch.slice(x, dim, start, end)：在指定轴切片；9223372036854775807 可读成“到末尾”。
# - torch.select(x, dim, i)：取该轴第 i 项并去掉这个轴。
# - reshape 改变维度组织；permute 改变轴顺序；二者配合决定数据的实际排列。
# - cat 沿已有轴拼接；stack 新增一个轴；unsqueeze 增加长度为 1 的轴。
# - forward1/forward2 是导出的方法名，不等于额外新增的网络层。
#
# 常量已从本地 3h 模型的 constants.pkl 类型信息和 constants/* 数据核对：
# c0=2；c1=6.283185307179586（2π）；c2=4；c3=1；c4=48（3×16）；
# c5=0.08164965809277261（约 1/sqrt(150)，注意力缩放系数）。
# 下文形状由这些常量和可见运算推导；未运行完整模型做中间特征采样。
# 其他子模块的卷积核、步幅等配置不全在本文件中，不凭名称猜测其精确值。
#
# 总体路线：
# 标准化物理场 + 时间标签
#   → 空间尺寸处理 → 拼接 6 个时间通道 → 卷积编码
#   → ViT / 14 层 Transformer → token 还原为空间特征
#   → 带跳跃连接的解码器 → 裁剪/恢复经度宽度 → 预测场
# 3h 是这个模型对应的预报间隔；3/6/9/12h 接力由外部 scheduler.py 负责。
# ================================================================

# 【整体模型】forward(x, time) 接收空间场和时间标签，输出预测空间场。
# 调用端 x 为 [B,54,H,W]，time 为 [B,3]；通常 B=1。
# 54 是物理变量/层组合的通道数。time 是当前输入场对应的时间，不是目标时效。
class TransUnet(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  init_conv : __torch__.torch.nn.modules.container.Sequential
  conv1 : __torch__.model.TransUnet.Bottleneck
  conv2 : __torch__.model.TransUnet.___torch_mangle_20.Bottleneck
  conv3 : __torch__.model.TransUnet.___torch_mangle_31.Bottleneck
  vit : __torch__.model.TransUnet.ViT
  project_patches_back : __torch__.torch.nn.modules.linear.___torch_mangle_210.Linear
  vit_conv : __torch__.model.TransUnet.SignleConv
  dec1 : __torch__.model.TransUnet.Up
  dec2 : __torch__.model.TransUnet.___torch_mangle_239.Up
  dec3 : __torch__.model.TransUnet.___torch_mangle_253.Up
  dec4 : __torch__.torch.nn.modules.container.___torch_mangle_255.Sequential

  def forward(self: __torch__.model.TransUnet.TransUnet,x: Tensor,time: Tensor) -> Tensor:

    dec4 = self.dec4
    dec3 = self.dec3
    dec2 = self.dec2
    dec1 = self.dec1
    
    vit_conv = self.vit_conv
    project_patches_back = self.project_patches_back
    vit = self.vit
    conv3 = self.conv3
    conv2 = self.conv2
    conv1 = self.conv1
    init_conv = self.init_conv

    # 阶段 1：统一空间尺寸。
    # _0=输入高度 H；先只把经度方向重采样到 500 列，高度不变。
    # nearest 表示最近邻重采样，属于模型内部预处理，不是预测时间推进。
    _0 = ops.prim.NumToTensor(torch.size(x, 2))
    x0 = torch.upsample_nearest2d(x, [int(_0), 500], None)

    # 下面三次 slice 都取整段，可合起来读成 x0[:, :, :, ...]。
    # left_pad 取最右侧 38 列，后面 right_pad 取最左侧 38 列。
    _1 = torch.slice(x0, 0, 0, 9223372036854775807)
    _2 = torch.slice(_1, 1, 0, 9223372036854775807)
    _3 = torch.slice(_2, 2, 0, 9223372036854775807)
    left_pad = torch.slice(_3, 3, -38, 9223372036854775807)
    _4 = torch.slice(x0, 0, 0, 9223372036854775807)
    _5 = torch.slice(_4, 1, 0, 9223372036854775807)
    _6 = torch.slice(_5, 2, 0, 9223372036854775807)
    right_pad = torch.slice(_6, 3, 0, 38)

    # 沿宽度轴拼接：[最右38列 | 原500列 | 最左38列]，宽度变成 576。
    # 这是经度方向的周期拼接，使左右边界可以接触到另一侧的数据。
    x1 = torch.cat([left_pad, x0, right_pad], 3)

    # 高度补到 576：rsub(H,576) 表示 576-H。
    # 上方补 floor((576-H)/2)，下方补剩余部分。
    # 此处没有指定填充值/模式的 torch.pad 是常量零填充；并不是纬度周期拼接。
    pad_height = torch.rsub(_0, 576)
    pad_top = torch.floor_divide(pad_height, CONSTANTS.c0)
    _7 = int(pad_top)
    _8 = int(pad_top)

    # pad_top = floor((576-H)/2)
    # pad_bottom = pad_height - pad_top
    # pad_bottom = 576 - pad_top
    pad_bottom = torch.sub(pad_height, pad_top)
    x2 = torch.pad(x1, [0, 0, _8, int(pad_bottom)])

    # 阶段 2：时间编码。
    # 外部调用端传入 [hour/24, dayofweek/7, dayofyear/365.25]。
    # 下面取出三列，每列都是 [B]。注意调用端 hour/24 没有加入 minute。
    # 每列经 cos(2πt)、sin(2πt) 转换，用成对周期特征表达时间。
    _9 = torch.slice(time, 0, 0, 9223372036854775807)
    time_of_day = torch.select(_9, 1, 0)
    _10 = torch.slice(time, 0, 0, 9223372036854775807)
    day_of_week = torch.select(_10, 1, 1)
    _11 = torch.slice(time, 0, 0, 9223372036854775807)
    day_of_year = torch.select(_11, 1, 2)
    _12 = torch.mul(time_of_day, CONSTANTS.c1)
    _13 = torch.cos(_12)
    _14 = torch.mul(day_of_week, CONSTANTS.c1)
    _15 = torch.cos(_14)
    _16 = torch.mul(day_of_year, CONSTANTS.c1)

    # 三个 [B] 向量 stack 成 [B,3]：时刻、星期、年内日期的余弦特征。
    cos_features = torch.stack([_13, _15, torch.cos(_16)], 1)

    _17 = torch.mul(time_of_day, CONSTANTS.c1)
    _18 = torch.sin(_17)
    _19 = torch.mul(day_of_week, CONSTANTS.c1)
    _20 = torch.sin(_19)
    _21 = torch.mul(day_of_year, CONSTANTS.c1)

    # 同样得到 [B,3] 的正弦特征。
    sin_features = torch.stack([_18, _20, torch.sin(_21)], 1)
    # [B,3] → [B,6]；顺序是三个 cos，接着三个 sin。
    time_features = torch.cat([cos_features, sin_features], 1)

    # 读取补齐后的场形状 [B,C,H,W]，这里 H=W=576。
    B = ops.prim.NumToTensor(torch.size(x2, 0))
    _22 = int(B)
    H = ops.prim.NumToTensor(torch.size(x2, 2))
    _23 = int(H)
    W = ops.prim.NumToTensor(torch.size(x2, 3))
    _24 = int(W)
    _25 = torch.slice(time_features, 0, 0, 9223372036854775807)
    _26 = torch.slice(_25, 1, 0, 9223372036854775807)
    
    # [B,6] → [B,6,1,1]，再 expand 成 [B,6,H,W]。
    # 每个样本的同一时间特征在所有网格点相同；expand 表达广播，不是学习操作。
    _27 = torch.unsqueeze(torch.unsqueeze(_26, 2), 3)
    time_features0 = torch.expand(_27, [_22, 6, _23, _24])

    # 沿通道拼接：54 个物理通道 + 6 个时间通道 → [B,60,576,576]。
    # 这里“时间进入网络”的具体方式就是增加空间常量通道。
    input = torch.cat([x2, time_features0], 1)

    # 阶段 3：卷积编码并保留中间特征。
    # _28=初始卷积特征；_29=conv1 输出；_30=conv2 输出。
    # 这三份特征稍后送入解码器，构成 U-Net 式跳跃连接。
    # 精确通道数/步幅应结合对应子模块配置核对，不能只根据 conv 名称判断。
#     | 操作 | 通道变化 | 空间尺寸变化 | 原因 |
    # |---|---|---|---|
    # | `init_conv` | 60 → 640 | 576×576 → 288×288 | 7×7 卷积，步幅 2，填充 3 |
    # | `conv1` | 640 → 1280 | 288×288 → 144×144 | 模块内的 3×3 卷积步幅为 2 |
    # | `conv2` | 1280 → 2560 | 144×144 → 72×72 | 模块内的 3×3 卷积步幅为 2 |
    # input.shape = [B, 60, 576, 576]
    _28 = (init_conv).forward(input, )
    # [B, 640, 288, 288]
    _29 = (conv1).forward(_28, )
    # [B, 1280, 144, 144]
    _30 = (conv2).forward(_29, )
    # [B, 2560, 72, 72]
    
    # conv3 的输出先进入 ViT；Transformer 处理的是卷积特征，而非直接处理原始54通道场。
    # 经过外层 conv3 模块, features.shape = [B, 5120, 36, 36]
    features = conv3.forward(_30)

    # ViT 返回已去掉 cls_token 的空间 token 序列 [B,N,D]。
    # 由下方还原操作可见，此导出路径要求 N=18×18=324。
    _31 = (vit).forward(features, )

    # 阶段 4：将每个 token 投影成可还原为空间特征的向量。
    # 设输出最后一维为 4*C_back，则 _32 为 [B,324,4*C_back]。
    # 这一步是 Linear，不是 reshape；它包含可学习参数。
    _32 = (project_patches_back).forward(_31, )
    element = ops.prim.NumToTensor(torch.size(_32, 0))
    _33 = int(element)
    _34 = ops.prim.NumToTensor(torch.size(_32, 2))

    # C_back = _32.shape[2] // 4；c3=1，后面的乘1只是导出展开。
    element0 = torch.floor_divide(_34, CONSTANTS.c2)
    _35 = int(element0)
    _36 = int(torch.mul(element, CONSTANTS.c3))
    _37 = int(torch.mul(element0, CONSTANTS.c3))

    # [B,324,4*C_back] → [B,18,18,2,2,C_back]
    # → permute 为 [B,C_back,2,18,2,18] → [B,C_back,36,36]。
    # 注意轴排列是“2,18”，不是常见相邻 patch 还原中的“18,2”；必须照实际运算理解。
    x3 = torch.reshape(_32, [_33, 18, 18, 2, 2, _35])
    x4 = torch.permute(x3, [0, 5, 3, 1, 4, 2])
    input0 = torch.reshape(x4, [_36, _37, 36, 36])

    # 阶段 5：解码恢复空间细节。
    # dec1(深层特征, conv2特征)；dec2(上一结果, conv1特征)；
    # dec3(上一结果, init_conv特征)。Up 类里可看到两路按通道拼接。
    # 跳跃连接给解码端提供编码阶段保留下来的空间信息。
    # vit_conv：SingleConv，1×1 卷积，通道数从 C_back → 2560，便于与 dec3 拼接。
    _38 = (dec1).forward((vit_conv).forward(input0, ), _30, )

    _39 = (dec3).forward((dec2).forward(_38, _29, ), _28, )

    # 最后一个 Sequential 生成输出特征/预测通道；其内部细节需看对应导出子模块。
    _40 = (dec4).forward(_39, )


    # 阶段 6：去掉前面添加的边界。
    # x5 约等于 _40[:, :, pad_top:-pad_bottom, :]。
    # 这是实际导出的负索引写法；若 pad_bottom=0，-0 就是0，不能把它当作通用尺寸处理器。
    _41 = int(torch.neg(pad_bottom))
    _42 = torch.slice(_40, 0, 0, 9223372036854775807)
    _43 = torch.slice(_42, 1, 0, 9223372036854775807)
    x5 = torch.slice(torch.slice(_43, 2, _7, _41), 3, 0, 9223372036854775807)
    _44 = torch.slice(x5, 0, 0, 9223372036854775807)
    _45 = torch.slice(_44, 1, 0, 9223372036854775807)
    _46 = torch.slice(_45, 2, 0, 9223372036854775807)

    # 去掉左右各 38 列周期填充，宽度回到 500。
    x6 = torch.slice(_46, 3, 38, -38)
    _47 = ops.prim.NumToTensor(torch.size(x6, 2))

    # 再把宽度重采样到固定 576；并非动态恢复任意输入宽度。
    # 外部流程把输出当作标准化场继续接力，最后才反标准化为物理量。
    # 本 forward 没有显式执行“最终输出 + 原始输入场”的残差预测操作。
    _48 = torch.upsample_nearest2d(x6, [int(_47), 576], None)
    return _48

  
# 【卷积残差块】主分支做多次卷积；捷径分支变换输入以匹配形状，最后相加。
# 注意 bn1/bn2/bn3 的实际类型是 InstanceNorm2d，不是 BatchNorm2d。
# ___torch_mangle_3.Conv2d：二维卷积层，会改变空间尺寸
# ___torch_mangle_4.InstanceNorm2d：实例归一化层，它不通过卷积核提取局部模式，也不混合不同通道，而是调整各通道特征图的尺度

class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  downsample : __torch__.torch.nn.modules.container.___torch_mangle_2.Sequential
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_3.Conv2d
  bn1 : __torch__.torch.nn.modules.instancenorm.___torch_mangle_4.InstanceNorm2d
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_5.Conv2d
  bn2 : __torch__.torch.nn.modules.instancenorm.___torch_mangle_6.InstanceNorm2d
  conv3 : __torch__.torch.nn.modules.conv.___torch_mangle_7.Conv2d
  bn3 : __torch__.torch.nn.modules.instancenorm.___torch_mangle_8.InstanceNorm2d
  relu : __torch__.torch.nn.modules.activation.___torch_mangle_9.ReLU
  def forward(self: __torch__.model.TransUnet.Bottleneck,
    argument_1: Tensor) -> Tensor:
    downsample = self.downsample
    bn3 = self.bn3
    conv3 = self.conv3
    bn2 = self.bn2
    conv2 = self.conv2
    relu = self.relu
    bn1 = self.bn1
    conv1 = self.conv1

    # 主分支：Conv1 → InstanceNorm1 → ReLU → Conv2 → InstanceNorm2
    # → ReLU → Conv3 → InstanceNorm3。
    _49 = (bn1).forward((conv1).forward(argument_1, ), )
    _50 = (conv2).forward((relu).forward(_49, ), )
    _51 = (relu).forward1((bn2).forward(_50, ), )
    _52 = (bn3).forward((conv3).forward(_51, ), )

    # 残差相加：主分支结果 + 捷径分支 downsample(input)。
    # 两路必须形状一致；add_ 是原地加法。这里的“残差连接”不是预报误差校正模型。
    # 【？】为什么要用下采样，而不是直接与argument_1相加？
    # 因为 argument_1 经过几次卷积之后的形状已经不是 [B,C,Hf,Wf] 了，需要下采样到 [B,C,Hf,Wf] 才能相加。
    input = torch.add_(_52, (downsample).forward(argument_1, ))

    # ReLU(x)=max(0,x)
    return (relu).forward2(input, )


# 【空间特征 → token → Transformer】这是你优先学习的模块。
# 输入来自 conv3，记为 [B,C,Hf,Wf]；输出是 [B,N,D]。
# cls_token 和 pos_emb1D 是可学习参数，project_patches 是可学习线性投影。
# 这里既没有图像分类输出头，也不直接输出气溶胶浓度。
class ViT(Module):
  __parameters__ = ["cls_token", "pos_emb1D", ]
  __buffers__ = []
  cls_token : Tensor
  pos_emb1D : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  project_patches : __torch__.torch.nn.modules.linear.Linear
  emb_dropout : __torch__.torch.nn.modules.dropout.Dropout
  transformer : __torch__.model.TransUnet.TransformerEncoder
  def forward(self: __torch__.model.TransUnet.ViT, argument_1: Tensor) -> Tensor:
    transformer = self.transformer
    emb_dropout = self.emb_dropout
    pos_emb1D = self.pos_emb1D
    cls_token = self.cls_token
    project_patches = self.project_patches

    # argument_1 是 conv3 输出的空间特征 [B,C,Hf,Wf]，其中 Hf=Wf=36。
    # 读取 B、C、Hf、Wf。下面 c0=2，因此 nH=Hf/2，nW=Wf/2。
    # 适用输入需要能满足这些整数 reshape；该导出路径对应 Hf=Wf=36。
    size_0 = ops.prim.NumToTensor(torch.size(argument_1, 0))
    _53 = int(size_0)
    size_1 = ops.prim.NumToTensor(torch.size(argument_1, 1))
    _54 = int(size_1)
    size_2 = ops.prim.NumToTensor(torch.size(argument_1, 2))
    size_3 = ops.prim.NumToTensor(torch.size(argument_1, 3))

    # size_2/2
    element2 = torch.floor_divide(size_2, CONSTANTS.c0)
    _57 = int(element2)
    # size_3/2 
    element3 = torch.floor_divide(size_3, CONSTANTS.c0)
    _58 = int(element3)

    # c3=1：_59=B；_60=nH*nW=N；_61=4*C。
    # 后面每个 token 用4个空间位置、每个位置C个通道组成。
    # size_0*c3 
    _59 = int(torch.mul(size_0, CONSTANTS.c3))
    result = torch.mul(element2, CONSTANTS.c3)
    _60 = int(torch.mul_(result, element3))
    _61 = int(torch.mul(size_1, CONSTANTS.c2))

    # 关键：此处不是把相邻 2×2 小块直接拼成 token！
    # [B,C,Hf,Wf] → [B,C,2,nH,2,nW]。
    # 原坐标对应 row=a*nH+i，col=b*nW+j，其中 a,b∈{0,1}。
    # 下方 permute 令 (i,j) 成为 token 位置，(a,b,C) 成为 token 特征。
    # 所以同一 token 取四个象限中相对位置相同的点；不要套用标准相邻 patch 的解释。
    # 例：Hf=Wf=4、C=1，输入按行是0..15，则首个 token 是 [0,2,8,10]。
    x = torch.reshape(argument_1, [_53, _54, 2, _57, 2, _58])

    # 轴重排：[B,C,2,nH,2,nW] → [B,nH,nW,2,2,C]。[B, 18, 18, 2, 2, 5120]
    x7 = torch.permute(x, [0, 3, 5, 2, 4, 1])
    
    # 合并空间索引和特征索引，得到 [B,N,4*C]。
    # 在36×36特征图上，N=18×18=324。
    img_patches = torch.reshape(x7, [_59, _60, _61])
    _62 = ops.prim.NumToTensor(torch.size(img_patches, 0))
    tokens = ops.prim.NumToTensor(torch.size(img_patches, 1))

    # 对每个 token 的 4*C 个特征做相同的线性变换：4*C → D。
    # 输出 [B,N,D]；这里尚未让不同 token 之间交换信息。
    _63 = (project_patches).forward(img_patches, )

    # 准备 cls_token 的批量扩展。下面较长的 size/乘法/reshape 是导出后的展开。
    # 按该参数的预期 [1,1,D] 形状，可理解为把同一个可学习 token 扩展给 B 个样本。
    other = ops.prim.NumToTensor(torch.size(cls_token, 0))
    element4 = torch.floor_divide(_62, other)
    _64 = int(element4)
    element5 = ops.prim.NumToTensor(torch.size(cls_token, 0))
    element6 = ops.prim.NumToTensor(torch.size(cls_token, 1))
    element7 = ops.prim.NumToTensor(torch.size(cls_token, 2))
    result0 = torch.mul(element5, CONSTANTS.c3)
    _65 = int(torch.mul_(result0, element4))
    _66 = int(torch.mul(element6, CONSTANTS.c3))
    _67 = int(torch.mul(element7, CONSTANTS.c3))

    # 先添加用于批量扩展的轴，再 expand、reshape；没有为每个样本学习独立参数。
    x8 = torch.unsqueeze(cls_token, 1)
    x9 = torch.expand(x8, [-1, _64, -1, -1])
    _68 = [torch.reshape(x9, [_65, _66, _67]), _63]

    # 沿序列轴把 cls_token 放在最前：[B,1,D]+[B,N,D] → [B,N+1,D]。
    # 它会参加注意力计算；虽然最后被删掉，也可能间接影响其他 token。
    img_patches0 = torch.cat(_68, 1)
    # 取前 N+1 个位置向量，形状 [N+1,D]，与整个 batch 广播相加。
    # 位置编码帮助网络区分序列位置；这里是可学习 pos_emb1D，别与前面的正余弦时间编码混淆。
    _69 = int(torch.add(tokens, CONSTANTS.c3))
    _70 = torch.slice(torch.slice(pos_emb1D, 0, 0, _69), 1, 0, 9223372036854775807)
    input = torch.add(img_patches0, _70)

    # 经过 embedding dropout，再经过14层 Transformer。
    # emb_dropout 用于在训练时随机将一部分输入元素置零，以减少过拟合
    # 序列长度和 token 特征维数在这些 block 中保持不变：[B,N+1,D]。
    # 推理调用端使用 eval 模式；实际 dropout 行为还取决于导出子模块中的训练标志。
    _71 = (transformer).forward((emb_dropout).forward(input, ), )
    _72 = torch.slice(_71, 0, 0, 9223372036854775807)

    # torch.slice(张量, 维度编号, 开始位置, 结束位置)
    # 删除序列第0项 cls_token，保留所有空间 token：[B,N,D]。
    # 此时尚未还原为二维图，外层 project_patches_back 和 reshape 才负责还原。
    _73 = torch.slice(_72, 1, 1, 9223372036854775807)
    input1 = torch.slice(_73, 2, 0, 9223372036854775807)
    return input1
  
# 【14层串联】layers[0] 到 layers[13] 依次执行。
# 可以概念化为：for layer in self.layers: x = layer(x)。
# 下面先取13再取12只是取引用的顺序，不是实际执行顺序。
class TransformerEncoder(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  layers : __torch__.torch.nn.modules.container.ModuleList
  def forward(self: __torch__.model.TransUnet.TransformerEncoder,
    argument_1: Tensor) -> Tensor:
    layers = self.layers
    _13 = getattr(layers, "13")
    layers0 = self.layers
    _12 = getattr(layers0, "12")
    layers1 = self.layers
    _11 = getattr(layers1, "11")
    layers2 = self.layers
    _10 = getattr(layers2, "10")
    layers3 = self.layers
    _9 = getattr(layers3, "9")
    layers4 = self.layers
    _8 = getattr(layers4, "8")
    layers5 = self.layers
    _7 = getattr(layers5, "7")
    layers6 = self.layers
    _6 = getattr(layers6, "6")
    layers7 = self.layers
    _5 = getattr(layers7, "5")
    layers8 = self.layers
    _4 = getattr(layers8, "4")
    layers9 = self.layers
    _3 = getattr(layers9, "3")
    layers10 = self.layers
    _2 = getattr(layers10, "2")
    layers11 = self.layers
    _1 = getattr(layers11, "1")
    layers12 = self.layers
    _0 = getattr(layers12, "0")
    # 嵌套调用从内到外：先第0层，再第1层；随后依次2、3、4……13。
    _74 = (_1).forward((_0).forward(argument_1, ), )
    _75 = (_4).forward((_3).forward((_2).forward(_74, ), ), )
    _76 = (_7).forward((_6).forward((_5).forward(_75, ), ), )
    _77 = (_10).forward((_9).forward((_8).forward(_76, ), ), )
    _78 = (_12).forward((_11).forward(_77, ), )
    return (_13).forward(_78, )

  
# 【一个 Transformer 层】由注意力子层和前馈子层组成。
# 此处是 Post-LN（相加之后归一化）：
#   z = LayerNorm(x + Dropout(MHSA(x)))
#   y = LayerNorm(z + FFN(z))
# 两条加法都是残差连接，要求两端的 [B,L,D] 形状相同。
# FFN 是下面名为 linear 的 Sequential，不等于仅一层 Linear；内部结构需看其定义。
class TransformerBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  mhsa : __torch__.model.TransUnet.MultiHeadSelfAttention
  drop : __torch__.torch.nn.modules.dropout.___torch_mangle_34.Dropout
  norm_1 : __torch__.torch.nn.modules.normalization.LayerNorm
  norm_2 : __torch__.torch.nn.modules.normalization.___torch_mangle_35.LayerNorm
  linear : __torch__.torch.nn.modules.container.___torch_mangle_40.Sequential
  def forward(self: __torch__.model.TransUnet.TransformerBlock,
    argument_1: Tensor) -> Tensor:
    norm_2 = self.norm_2
    linear = self.linear
    norm_1 = self.norm_1
    drop = self.drop
    mhsa = self.mhsa

    # 注意力在不同 token 之间交换信息，之后经过 dropout。
    # 输出 [B,L,D]。
    _79 = (drop).forward((mhsa).forward(argument_1, ), )

    # 第一条残差：把输入 x 加回来
    # 更新后的表示 = 原来的表示 + 根据全序列信息计算出的更新
    input = torch.add(_79, argument_1)

    # 归一化：减去均值、再除以标准差，使每个 token 的特征分布更稳定。
    _80 = (norm_1).forward(input, )

    # 前馈网络对各 token 应用共享的特征变换
    _81 = (linear).forward(_80, )

    # 第二条残差：把前馈网络的输出加回来
    input2 = torch.add(_81, _80)

    # 归一化：减去均值、再除以标准差，使每个 token 的特征分布更稳定。
    return (norm_2).forward(input2, )
# 【多头自注意力】输入 X=[B,L,D]，L 包括 cls_token。
# Q（query）表达当前位置用什么特征匹配别人；K（key）提供被匹配特征；
# V（value）提供要被加权汇总的信息。这是计算角色，不是具体气象变量的固定对应。
# 每头：A=softmax(Q @ K^T * c5)，Y=A @ V；再合并各头并做输出投影。
# 此文件可见头数 h=16；c5≈1/sqrt(150)，对应通常的每头维数缩放形式。
# 确切投影宽度可再结合权重形状核对，不单凭缩放值代替权重证据。
# 本实现没有可见的因果掩码：这里是空间 token 的相互注意，不是时间自回归掩码。
class MultiHeadSelfAttention(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  to_qvk : __torch__.torch.nn.modules.linear.___torch_mangle_32.Linear
  W_0 : __torch__.torch.nn.modules.linear.___torch_mangle_33.Linear
  def forward(self: __torch__.model.TransUnet.MultiHeadSelfAttention,
    argument_1: Tensor) -> Tensor:
    W_0 = self.W_0
    to_qvk = self.to_qvk

    # 一次线性投影生成供 Q/K/V 使用的特征，总宽度是 3*h*d。
    # 虽然属性名字拼作 to_qvk，后面要按实际运算角色判断 Q、K、V 的顺序。
    _81 = (to_qvk).forward(argument_1, )

    element = ops.prim.NumToTensor(torch.size(_81, 0))
    _82 = int(element)
    element8 = ops.prim.NumToTensor(torch.size(_81, 1))
    _83 = int(element8)
    _84 = ops.prim.NumToTensor(torch.size(_81, 2))

    # c4=48=3×16，所以每头维数 d = 投影总宽度/48。
    element9 = torch.floor_divide(_84, CONSTANTS.c4)

    # [B,L,3*h*d] → [B,L,d,3,16]。
    # 注意这里的排列是 d,3,h；不要直接换成常见教程里的 3,h,d，权重分组会不同。
    x = torch.reshape(_81, [_82, _83, int(element9), 3, 16])

    # 变为 [3,B,16,L,d]；第0轴用于区分 Q/K/V。
    qkv_rearranged = torch.permute(x, [3, 0, 4, 1, 2])

    # _85=Q，_86=K，_87=V，三者均为 [B,16,L,d]。
    # 这是由后面“前两者点积、第三者被加权”的用法确定的。
    _85 = torch.select(qkv_rearranged, 0, 0)
    _86 = torch.select(qkv_rearranged, 0, 1)
    _87 = torch.select(qkv_rearranged, 0, 2)

    # 对特征维 d 求和：score[b,h,i,j] = sum_d Q[b,h,i,d]*K[b,h,j,d]。
    # 省略号代表共同的 B、h 轴；结果 [B,16,L,L]。
    # i 是当前查询位置，j 是可读取的位置。它不是通道×通道矩阵。
    _88 = torch.einsum("... i d , ... j d -> ... i j", [_85, _86])

    # 乘缩放常量 c5，控制点积数值尺度，避免 softmax 过早变得极端。
    scaled_dot_prod = torch.mul(_88, CONSTANTS.c5)
    
    # softmax 沿最后的 j 轴计算：每个查询 i 对所有键位置 j 的权重和为1。
    # _89 只是 [注意力权重 A, V] 这个列表，并不是最终输出。
    _89 = [torch.softmax(scaled_dot_prod, -1), _87]

    # 加权汇总：Y[b,h,i,d] = sum_j A[b,h,i,j]*V[b,h,j,d]。
    # 结果 [B,16,L,d]。每个位置从全序列收集信息，但输出仍有 L 个位置。
    x10 = torch.einsum("... i j , ... j d -> ... i d", _89)
    
    # 读取 B、h、L、d；后续 c3=1 的乘法可直接略读。
    element10 = ops.prim.NumToTensor(torch.size(x10, 0))
    element11 = ops.prim.NumToTensor(torch.size(x10, 1))
    element12 = ops.prim.NumToTensor(torch.size(x10, 2))
    element13 = ops.prim.NumToTensor(torch.size(x10, 3))
    B = int(torch.mul(element10, CONSTANTS.c3))
    h = int(torch.mul(element12, CONSTANTS.c3))
    L = int(torch.mul(element11, CONSTANTS.c3))
    d = int(torch.mul(element13, CONSTANTS.c3))

    # [B,h,L,d] → [B,L,h,d]，随后合并 h*d。
    x11 = torch.permute(x10, [0, 2, 1, 3])

    # 得到 [B,L,h*d]，把16个头的结果拼起来。
    input = torch.reshape(x11, [B, L, h * d])

    # 最后用可学习线性层混合各头的信息，输出投影回 block 所需的特征宽度 D。
    # 输出 [B,L,D]。
    return (W_0).forward(input, )
# 【卷积包装模块】SignleConv 是导出名称的原拼写，保留不改。
# 这里仅调用内部 Sequential；不能从这个包装类看全卷积/归一化/激活配置。
class SignleConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.container.___torch_mangle_214.Sequential
  def forward(self: __torch__.model.TransUnet.SignleConv,
    input: Tensor) -> Tensor:
    conv = self.conv
    return (conv).forward(input, )

  
# 【带跳跃连接的上采样模块】
# argument_1：来自更深层/前一级解码器的特征；argument_2：编码器保存的特征。
# 先对 argument_1 做转置卷积上采样，再与 argument_2 按通道拼接。
# 转置卷积有可学习参数，不等于前面 nearest 的固定重采样。
class Up(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  up : __torch__.torch.nn.modules.conv.ConvTranspose2d
  conv : __torch__.model.TransUnet.DoubleConv
  def forward(self: __torch__.model.TransUnet.Up,
    argument_1: Tensor,
    argument_2: Tensor) -> Tensor:
    conv = self.conv
    up = self.up
    # 先计算上采样，再准备两路特征列表。二者空间 H/W 必须匹配。
    _93 = [argument_2, (up).forward(argument_1, )]
    # 沿通道拼接，而非逐元素相加：[B,C_skip,H,W]+[B,C_up,H,W]

    # → [B,C_skip+C_up,H,W]，随后 conv 融合两路特征。
    input = torch.cat(_93, 1)
    return (conv).forward(input, )

  
# 【解码端卷积包装】这里的 forward 把输入交给内部 Sequential。
# 类名提示双卷积设计，但核大小、通道数、激活等应查该 Sequential 的实际定义。
class DoubleConv(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.container.___torch_mangle_225.Sequential
  def forward(self: __torch__.model.TransUnet.DoubleConv,
    input: Tensor) -> Tensor:
    conv = self.conv
    return (conv).forward(input, )
