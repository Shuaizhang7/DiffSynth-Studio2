## 任务：更改/data/shuaizhang/DiffSynth-Studio中/data/shuaizhang/DiffSynth-Studio/diffsynth/models/wan_video_camera_controller.py的相机位姿控制方式

步骤1 读取/data/shuaizhang/DiffSynth-Studio/diffsynth/models/wan_video_camera_controller.py 中是如何进行相机位姿控制的

**分析结果：**

### 当前相机位姿控制方式：

1. **相机坐标格式**：每个坐标是包含19个元素的列表
   - `origin = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0)`
   - 索引4-5: `cx, cy` (归一化相机中心，0.5, 0.5)
   - 索引6-18: 4x4矩阵(展开为18个值)，即 w2c (world-to-camera) 矩阵

2. **`generate_camera_coordinates` 函数**：生成相机轨迹
   - 输入：direction (方向), length (帧数), speed (速度), origin (起始位姿)
   - 通过修改 w2c 矩阵元素来移动相机：
     - Left/Right: 修改索引9 (X平移)
     - Up/Down: 修改索引13 (Y平移)
     - In/Out: 修改索引18 (Z平移/缩放)

3. **`process_pose_file` 函数**：将相机参数转换为 Plucker embedding
   - 解析 Camera 对象（内参 fx, fy, cx, cy 和 w2c 矩阵）
   - 调整宽高比
   - 计算相对位姿 `get_relative_pose`
   - 调用 `ray_condition` 生成 Plucker embedding

4. **Plucker embedding**：用于表示相机射线的空间位置
   - 公式：`plucker = [rays_dxo, rays_d]`
   - `rays_dxo = rays_o × rays_d` (射线原点与方向的叉积)
   - `rays_d` (射线方向)

---

### 步骤2：修改相机位姿控制方式以支持 REALESTATE10K 数据集

**修改内容：**

1. **REALESTATE10K 位姿文件格式**：
   - 第一行：YouTube URL
   - 后续每行：`timestamp fx fy cx cy w2c_row0 w2c_row1 w2c_row2` (17个值)

2. **修改 `generate_camera_coordinates` 函数**：
   - 新增参数 `pose_file_path`：位姿文件路径
   - 新增参数 `num_frames`：读取帧数
   - 当 `pose_file_path` 不为空时，从文件读取位姿

3. **新增 `load_camera_poses_from_file` 函数**：
   - 从 REALESTATE10K 格式文件读取相机位姿
   - 跳过第一行 URL，解析剩余行

4. **修改 `Camera` 类**：
   - 支持两种格式：
     - 19元素格式（原始）：w2c 从索引7开始
     - 17元素格式（REALESTATE10K）：w2c 从索引5开始

**使用示例：**
```python
# 原有方式（基于方向生成）
coordinates = generate_camera_coordinates(direction="Left", length=16, speed=1/54)

# 新方式（从 REALESTATE10K 文件读取）
coordinates = generate_camera_coordinates(pose_file_path="/path/to/pose.txt", num_frames=16)
```

---

### 步骤3：修改推理脚本和 Pipeline 支持位姿文件

**修改内容：**

1. **修改 `SimpleAdapter.process_camera_coordinates`** ([wan_video_camera_controller.py:46-84](diffsynth/models/wan_video_camera_controller.py#L46-L84))
   - 新增 `pose_file_path` 和 `num_frames` 参数
   - 当 `pose_file_path` 不为空时，从文件读取位姿

2. **修改 `WanVideoUnit_FunCameraControl`** ([wan_video.py:550-565](diffsynth/pipelines/wan_video.py#L550-L565))
   - 添加 `camera_control_pose_file` 到 `input_params`
   - 修改 `process` 方法支持从文件读取

3. **修改 `WanVideoPipeline.__call__`** ([wan_video.py:195-199](diffsynth/pipelines/wan_video.py#L195-L199))
   - 新增 `camera_control_pose_file` 参数

4. **修改推理脚本** ([Wan2.2-Fun-A14B-Control-Camera.py:36-43](examples/wanvideo/model_inference/Wan2.2-Fun-A14B-Control-Camera.py#L36-L43))
   - 添加使用位姿文件的示例

**使用方式：**
```python
# 基于方向（原有方式）
video = pipe(
    prompt="...",
    input_image=input_image,
    camera_control_direction="Left", camera_control_speed=0.01,
)

# 从 REALESTATE10K 位姿文件读取
video = pipe(
    prompt="...",
    input_image=input_image,
    camera_control_pose_file="/path/to/pose.txt",
)
```


## 2026/01/12

### 帮我写一个python脚本读取Wan1.3B模型权重
权重的地址为：https://modelscope.cn/models/PAI/Wan2.1-Fun-V1.1-1.3B-Control/files

**完成状态：已创建推理脚本**

创建了脚本文件：[Wan2.1-Fun-V1.1-1.3B-Control-LoadWeights.py](examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-LoadWeights.py)

**脚本功能：**
1. 从 ModelScope 加载 Wan1.3B-Control 模型权重
2. 支持配置数据类型（bfloat16）和运行设备
3. 提供模型组件加载状态检查
4. 显示模型参数量统计

**使用方法：**
```python
from Wan2.1-Fun-V1.1-1.3B-Control-LoadWeights import load_wan_1_3b_model

pipe = load_wan_1_3b_model(torch_dtype=torch.bfloat16, device="cuda")
```

**模型文件列表：**
- `diffusion_pytorch_model*.safetensors` - 扩散模型主权重
- `models_t5_umt5-xxl-enc-bf16.pth` - T5 文本编码器
- `Wan2.1_VAE.pth` - VAE 模型
- `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` - CLIP 图像编码器


### 目前已经下载好了权重，请按照说明进行推理测试

运行脚本/data2/shuaizhang/DiffSynth-Studio2/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py

### 目前已经跑通了基础的脚本，请你根据上面写的内容，将/data2/shuaizhang/DiffSynth-Studio2/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py脚本，先备份，再修改为提供camera_control_pose_file控制相机位姿的脚本，camera_control_pose_file的地址为：/data2/shuaizhang/DiffSynth-Studio2/pose/00a5a2af678f37d5.txt

**完成状态：已备份并添加位姿文件测试**

1. **备份文件**：[Wan2.1-Fun-V1.1-1.3B-Control-Camera.py.backup](examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py.backup)

2. **修改内容**：
   - 添加了使用 REALESTATE10K 位姿文件的测试用例
   - 新增 `camera_control_pose_file="pose/00a5a2af678f37d5.txt"` 参数
   - 输出文件：`video_pose_file_Wan2.1-Fun-V1.1-1.3B-Control-Camera.mp4`

3. **使用方式**：
   ```python
   video = pipe(
       prompt="...",
       negative_prompt="...",
       seed=0, tiled=True,
       input_image=input_image,
       camera_control_pose_file="pose/00a5a2af678f37d5.txt",
   )
   ```

**运行命令**：
```bash
cd /data2/shuaizhang/DiffSynth-Studio2
python examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py
```

## 2026/01/13
### 下载样例数据集，地址在https://modelscope.cn/datasets/DiffSynth-Studio/example_video_dataset


modelscope download --dataset DiffSynth-Studio/example_video_dataset

### 下载数据完成

## 2026/01/16

目前我完成了模型的训练，但是模型只有训练脚本，在训练过程中没有可视化的验证，我想在训练的过程中，同时隔一段训练验证保存的checkpoint模型。

验证模型的脚本可以参考/data/vepfs/users/shuaizhang/DiffSynth-Studio2/examples/wanvideo/model_inference/Wan2.1-Fun-V1.1-1.3B-Control-Camera.py

原本的训练脚本为：/data/vepfs/users/shuaizhang/DiffSynth-Studio2/examples/wanvideo/model_training/train.py

你可以参考：/data/vepfs/users/shuaizhang/DualCamCtrl/examples/wanvideo/model_training/train_with_accelerate.py

过程比较复杂，在你修改原本的训练脚本时，先备份一下，一步一步来，步骤记录到本文档中


## 2026/01/17

### 完成了训练脚本的修改，添加了验证和 checkpoint 保存功能

**完成状态：已修改**

1. **备份文件**：[train.py.backup](examples/wanvideo/model_training/train.py.backup)

2. **修改内容**：
   - 添加了 `ModelLogger` 类 - 管理 checkpoint 保存和加载
   - 添加了 `validate()` 方法到 `WanTrainingModule` - 生成验证视频
   - 添加了验证相关参数到 parser
   - 重写了训练循环，添加验证逻辑

3. **新增参数**：
   - `--validate_step`: 每 N 步验证和保存 checkpoint（默认 500）
   - `--log_step`: 每 N 步打印 loss（默认 10）
   - `--init_validate`: 训练前是否验证（默认 False）
   - `--validate_batch`: 验证批次数（默认 1）
   - `--validation_dataset_metadata_path`: 验证数据集路径

4. **使用方式**：
   ```bash
   # 带验证的训练
   python examples/wanvideo/model_training/train.py \
       --output_path ./output \
       --dataset_base_path ./data \
       --dataset_metadata_path ./metadata/train.json \
       --validation_dataset_metadata_path ./metadata/val.json \
       --validate_step 500 \
       --init_validate \
       --num_epochs 10
   ```

5. **输出文件**：
   - Checkpoint: `output/checkpoint-step-{global_step}/`
   - 验证视频: `output/validation_results_50_inference_steps/video_{idx}_step_{global_step}.mp4`


## 20260118
### 目前验证时可视化出来的是灰度视频，请你帮我增加一个功能，额外保存一个颜色映射后的灰度图视频

参考：
colormap = np.array(cm.get_cmap("inferno").colors)
depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm

**完成状态：已添加颜色映射视频保存功能**

1. **备份文件**：[train.py.backup2](examples/wanvideo/model_training/train.py.backup2)

2. **修改内容**：
   - 添加 matplotlib 导入：`matplotlib.colors`, `matplotlib.cm`
   - 添加 `apply_colormap_to_video` 静态方法到 `WanTrainingModule`
   - 修改验证保存逻辑，同时保存原始视频和颜色映射版本

3. **新增 `apply_colormap_to_video` 方法**：
   - 将灰度视频转换为 RGB 颜色映射视频
   - 使用 `inferno` colormap
   - 支持多种输入格式：(T, H, W), (T, C, H, W)

4. **输出文件**：
   - 原始视频：`video_pose_0_step_500.mp4`
   - 颜色映射视频：`video_pose_0_step_500_colormap.mp4`


## 20260121
### 目前训练的过程中，位姿的控制方式还是通过CSV文件中给定的运动方向提示词，例如"left，right"等。请修改成支持 REALESTATE10K 数据集的格式，即直接输入相机位姿文件进行控制。

**完成状态：已修改为支持 REALESTATE10K 位姿文件格式**

1. **备份文件**：[train.py.backup3](examples/wanvideo/model_training/train.py.backup3)

2. **修改内容**：

   a. **导入相机控制器模块** ([train.py:7-10](examples/wanvideo/model_training/train.py#L7-L10))
   ```python
   from diffsynth.models.wan_video_camera_controller import (
       generate_camera_coordinates,
       process_pose_file,
   )
   ```

   b. **修改 `get_pipeline_inputs` 方法** ([train.py:214-229](examples/wanvideo/model_training/train.py#L214-L229))
   - 新增对 `pose_path` 字段的支持
   - 从位姿文件加载相机坐标并转换为 Plucker embedding
   - 将 plucker embedding 传递给 pipeline

   c. **修改 Pipeline 支持 `camera_control_plucker` 参数** ([wan_video.py:200](diffsynth/pipelines/wan_video.py#L200))
   - 添加 `camera_control_plucker` 参数到 `__call__` 方法
   - 修改 `WanVideoUnit_FunCameraControl` 支持预计算的 plucker embedding

3. **数据集格式**：
   - 新的 CSV 格式使用 `pose_path` 字段指向 REALESTATE10K 格式的位姿文件
   ```csv
   video_path,text,num_frames,height,width,flow,pose_path
   /path/to/video.mp4,Room,33,256,256,1,/path/to/pose.txt
   ```

4. **使用方式**：
   ```python
   # 训练脚本参数
   --dataset_metadata_path ./metadata/pose_train.csv
   ```

5. **修改的文件**：
   - [train.py](examples/wanvideo/model_training/train.py) - 添加位姿文件支持
   - [wan_video.py](diffsynth/pipelines/wan_video.py) - 添加 camera_control_plucker 参数

**向后兼容**：原有基于 `camera_control_direction` 的训练方式仍然支持，新旧格式可以共存。

### 20260121 (补充)
#### 验证时推理步数改为30步

**完成状态：已修改**

1. **修改内容**：
   - 将验证时的推理步数从 50 改为 30
   - 新增 `--validate_num_inference_steps` 参数，可自定义验证推理步数

2. **使用方式**：
   ```bash
   # 默认使用30步
   python train.py --validate_step 500

   # 自定义验证推理步数
   python train.py --validate_step 500 --validate_num_inference_steps 50
   ```