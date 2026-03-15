import cv2
import numpy as np
from PIL import Image


def generate_overlay_heatmap(img_path, heat_matrix):
    """
    将真实的特征热力矩阵插值并叠加到原图上。
    """
    try:
        # 读取原图并转为 RGB
        original = cv2.imread(img_path)
        if original is None:
            return img_path
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        h, w = original.shape[:2]

        # +++ 核心逻辑：将 14x14 的真实矩阵平滑放大到原图尺寸 +++
        heat_matrix_resized = cv2.resize(heat_matrix, (w, h), interpolation=cv2.INTER_CUBIC)

        # 映射为伪彩色 (JET色系：红色最高，蓝色最低)
        heat_matrix_color = np.uint8(255 * heat_matrix_resized)
        colormap = cv2.applyColorMap(heat_matrix_color, cv2.COLORMAP_JET)
        colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

        # 叠加原图与热力图 (alpha=0.6, beta=0.4 保证底片可见度)
        overlay = cv2.addWeighted(original, 0.6, colormap, 0.4, 0)

        # 保存为临时文件供 Streamlit 渲染
        temp_out = img_path.replace(".png", "_heat.png").replace(".jpg", "_heat.jpg").replace(".dcm", "_heat.jpg")
        Image.fromarray(overlay).save(temp_out)
        return temp_out
    except Exception as e:
        print(f"真实热力图生成失败: {e}")
        return img_path