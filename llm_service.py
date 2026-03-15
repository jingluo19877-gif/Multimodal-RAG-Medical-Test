import base64
import os
import cv2
import numpy as np
from ollama import Client
from PIL import Image


class MedicalLLMService:
    def __init__(self, remote_url):
        self.client = Client(host=remote_url)
        self.diagnostic_model = 'dcarrascosa/medgemma-1.5-4b-it:Q8_0'
        self.translator_model = 'qwen2.5:1.5b'

    def translate_to_chinese(self, english_text):
        """将英文文本翻译为专业中文"""
        if not english_text.strip():
            return ""

        prompt = (
            "You are a professional medical translator specializing in radiology reports. "
            "Translate the following English radiology report into Chinese.\n\n"
            "【Requirements】:\n"
            "1. **Preserve the exact structure**: Keep the headings "
            "**【Imaging Findings】** and **【Diagnosis】** unchanged (they will be automatically converted to "
            "**【影像学表现】** and **【诊断意见】** in Chinese).\n"
            "2. **Use precise Chinese medical terminology**: For example, 'consolidation' → '实变', "
            "'pleural effusion' → '胸腔积液', 'nodule' → '结节', 'pneumothorax' → '气胸'.\n"
            "3. **Maintain numerical accuracy**: Ensure measurements (e.g., '5 mm') are correctly translated "
            "as '5 mm' (keep units in mm/cm as appropriate).\n"
            "4. **Output only the translation** – do not add any explanations, comments, or extra text.\n"
            "5. **Ensure fluent and coherent Chinese** while strictly adhering to medical accuracy.\n\n"
            "English Report:\n"
            f"{english_text}\n\n"
            "Chinese Translation:"
        )

        response = self.client.chat(
            model=self.translator_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.1}
        )
        return response['message']['content'].strip()


    def translate_to_english(self, chinese_query):
        """用 Few-Shot 强约束，只允许输出翻译"""
        prompt = (
            "Task: Translate the medical query from Chinese to English. "
            "Rule: ONLY output the translated English term. NO explanations. NO medical advice. NO chat.\n\n"
            "Example 1:\nInput: 肺部有阴影\nOutput: Lung opacity\n\n"
            "Example 2:\nInput: 心脏肥大\nOutput: Cardiomegaly\n\n"
            f"Current Task:\nInput: {chinese_query}\nOutput:"
        )
        response = self.client.chat(model=self.translator_model, messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip()

    def stream_diagnosis(self, image_path, user_query_english, retrieved_context, roi_description=None):
        """
        流式生成英文诊断报告
        :param image_path: 全局影像路径
        :param user_query_english: 用户问题（英文翻译）
        :param retrieved_context: 检索到的相似病例报告文本（英文）
        :param roi_description: 用户圈选的区域描述（如"upper left lung field"），若无则为None
        """
        if not os.path.exists(image_path):
            yield f"Local path error: image file {image_path} not found."
            return

        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        roi_instruction = ""
        if roi_description:
            roi_instruction = f"【Region of Interest】: {roi_description}. Please pay special attention to this area and integrate with the global image.\n\n"

        combined_prompt = (
            "You are a senior radiologist at a top-tier hospital. Based on the provided chest X-ray image, the patient's chief complaint, and reference similar cases, generate a comprehensive radiology report in English. Focus on the following core elements:\n\n"
            f"{roi_instruction}"
            "【Core Assessment Areas】:\n"
            "1. **Heart and Mediastinum**: Size, shape, position; any signs of cardiomegaly, mediastinal widening, or hilar prominence (describe qualitatively, e.g., 'mildly enlarged', 'normal', 'prominent').\n"
            "2. **Lungs and Pleura**: Lung fields – any focal opacities (consolidation, nodule, mass), interstitial markings, atelectasis, hyperinflation; pleural abnormalities (effusion, pneumothorax, pleural thickening).\n"
            "3. **Thoracic Cage**: Bony structures – ribs, clavicles, spine – for fractures, lesions, or deformities.\n"
            "4. **Other Findings**: Diaphragm (elevation, free air), soft tissues, support devices (tubes, lines).\n\n"
            "【Detailed Reporting Instructions】:\n"
            "- **Normal findings**: Describe as 'unremarkable', 'clear', 'within normal limits' with sufficient detail (e.g., 'The heart size is normal, mediastinal contour is unremarkable, and lung fields are clear without focal consolidation').\n"
            "- **Abnormal findings**: For each abnormality, provide:\n"
            "   * **Location**: precise lobe/segment, zone (upper/middle/lower), or anatomical landmark.\n"
            "   * **Qualitative size**: use descriptive terms such as 'small', 'moderate', 'large', 'tiny' (avoid absolute measurements in mm/cm).\n"
            "   * **Morphology**: shape (round, oval, irregular), margin (smooth, spiculated, lobulated), density (solid, ground-glass, cavitary), and any associated features (calcification, air bronchograms, satellite lesions).\n"
            "   * **Extent**: for diffuse processes (e.g., 'interstitial thickening in both lower zones').\n"
            "   * **Comparison**: if relevant to prior exams (but only if available).\n"
            "- **Use precise medical terminology** (e.g., 'consolidation', 'effusion', 'pneumothorax', 'reticulation', 'atelectasis').\n\n"
            "【Report Format - Strictly Follow】:\n"
            "**【Imaging Findings】**\n"
            "(Detailed description of all observed features, organized by anatomical area, as above.)\n\n"
            "**【Diagnosis】**\n"
            "(Concise interpretation: list key findings, differential diagnoses if applicable, and specific recommendations based on clinical context. Avoid relying on specific measurements; use qualitative descriptors and clinical judgment, e.g., 'Likely community-acquired pneumonia; follow-up radiography after treatment recommended' or 'Indeterminate nodule with spiculated margins – recommend chest CT for further characterization'.)\n\n"
            "【Example Report – Emulate this level of detail】:\n"
            "**【Imaging Findings】**\n"
            "The cardiomediastinal silhouette is normal in size and contour. Lungs are clear without focal consolidation, nodule, or mass. No pleural effusion or pneumothorax is seen. The costophrenic angles are sharp. The bony thorax shows no acute fracture. Incidental note of mild degenerative changes of the thoracic spine.\n\n"
            "**【Diagnosis】**\n"
            "No acute intrathoracic abnormality.\n\n"
            "---Start Diagnosis---\n"
            f"Chief Complaint: {user_query_english}\n"
            f"Retrieved Similar Cases (highly similar to the region of interest):\n{retrieved_context}\n"
        )

        stream = self.client.chat(
            model=self.diagnostic_model,
            messages=[
                {
                    'role': 'user',
                    'content': combined_prompt,
                    'images': [image_base64]
                }
            ],
            stream=True,
            options={"num_predict": 4096, "temperature": 0.2}
        )

        for chunk in stream:
            yield chunk['message']['content']

    @staticmethod
    def generate_overlay_heatmap(img_path, heat_matrix=None):
        """
        将特征热力图叠加到原图上。
        heat_matrix: 理想情况下应由 BiomedCLIP 的自注意力层输出。
        此处提供平滑的高斯分布矩阵作为视觉演示占位。
        """
        try:
            # 读取原图并转为 RGB
            original = cv2.imread(img_path)
            if original is None:
                return img_path
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            # 如果没有传入真实的注意力矩阵，生成一个模拟的中心聚焦热力矩阵
            if heat_matrix is None:
                heat_matrix = np.zeros(original.shape[:2], dtype=np.float32)
                cv2.circle(heat_matrix, (original.shape[1] // 2, original.shape[0] // 2),
                           min(original.shape[:2]) // 3, 1.0, -1)
                heat_matrix = cv2.GaussianBlur(heat_matrix, (99, 99), 0)

            # 归一化并转为伪彩色
            heat_matrix = np.uint8(255 * heat_matrix)
            colormap = cv2.applyColorMap(heat_matrix, cv2.COLORMAP_JET)
            colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

            # 叠加原图与热力图 (alpha=0.6, beta=0.4)
            overlay = cv2.addWeighted(original, 0.6, colormap, 0.4, 0)

            # 保存为临时文件供 Streamlit 渲染
            temp_out = img_path.replace(".png", "_heat.png").replace(".jpg", "_heat.jpg")
            Image.fromarray(overlay).save(temp_out)
            return temp_out
        except Exception as e:
            return img_path  # 如果生成失败，降级返回原图