import os
import kagglehub
import shutil

# 1. 定义你想要存放数据的项目内路径
# 假设你想存放在项目根目录下的 data/iu_xray 文件夹
local_data_dir = os.path.join(os.getcwd(), "data", "iu_xray")

# 如果文件夹不存在则创建
if not os.path.exists(local_data_dir):
    os.makedirs(local_data_dir)
    print(f"已创建目录: {local_data_dir}")

# 2. 修改 kagglehub 的下载缓存路径（临时修改环境变量）
os.environ["KAGGLEHUB_CACHE"] = local_data_dir

# 3. 执行下载
print("正在从 Kaggle 下载 IU X-Ray 数据集到项目目录...")
# 注意：kagglehub 还是会根据数据集名创建子文件夹
downloaded_path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")

print(f"\n✅ 数据集已就绪！")
print(f"最终存放路径: {downloaded_path}")

# 4. (可选) 如果你想让结构更扁平，可以直接把里面的文件移出来
# 现在的路径通常是 project/data/iu_xray/datasets/raddar/...