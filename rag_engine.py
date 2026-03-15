import os
import pandas as pd
import torch
import numpy as np
import pydicom
from PIL import Image
import chromadb
from llama_index.core.schema import TextNode, ImageNode
from open_clip import create_model_from_pretrained, get_tokenizer


class MedicalRAGEngine:
    def __init__(self, db_path="./storage/medical_vectordb"):
        self.db_path = db_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 加载 BiomedCLIP (修复区：统一命名为 self.clip_model 并发送到计算设备)
        print("正在加载 BiomedCLIP 模型，请稍候...")
        self.clip_model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.clip_model.to(self.device)
        self.clip_model.eval()  # 切换到推理模式

        # 2. 初始化 ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="iu_xray_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def load_medical_image(self, img_path):
        """兼容处理 .dcm 和普通图片的读取"""
        if img_path.lower().endswith('.dcm'):
            # 读取 DICOM 并转换为 PIL Image
            dicom_data = pydicom.dcmread(img_path)
            pixel_array = dicom_data.pixel_array.astype(float)
            # 归一化到 0-255 范围
            pixel_array = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
            image = Image.fromarray(np.uint8(pixel_array)).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
        return image

    def extract_image_feature(self, img_path):
        """提取单张图片的向量"""
        pil_img = self.load_medical_image(img_path)
        image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feature = self.clip_model.encode_image(image_tensor)
            img_feature /= img_feature.norm(dim=-1, keepdim=True)

        return img_feature.cpu().numpy().tolist()[0]

    def build_index_from_csv(self, reports_csv, projections_csv, img_dir, max_samples=50, progress_callback=None,
                             log_callback=None):
        """联合两张表，构建节点并入库（增加增量去重校验逻辑）"""
        df_reports = pd.read_csv(reports_csv)
        df_projections = pd.read_csv(projections_csv)
        df_merged = pd.merge(df_projections, df_reports, on='uid', how='inner').head(max_samples)

        inserted_count = 0
        skipped_count = 0
        total_samples = len(df_merged)

        for index, row in df_merged.iterrows():
            uid = str(row['uid'])
            filename = str(row['filename'])
            findings = str(row.get('findings', ''))
            impression = str(row.get('impression', ''))

            findings = "" if findings == 'nan' else findings
            impression = "" if impression == 'nan' else impression
            report_text = f"Findings: {findings}\nImpression: {impression}".strip()

            img_path = os.path.join(img_dir, filename)

            if len(report_text) > 15 and os.path.exists(img_path):
                # +++ 核心新增：增量去重校验 +++
                # 通过 get 方法极速查询 ChromaDB 中是否已有该文件 ID
                existing_record = self.collection.get(ids=[filename])

                if existing_record and existing_record['ids'] and len(existing_record['ids']) > 0:
                    skipped_count += 1
                    if log_callback:
                        log_callback(f"⏭️ 命中缓存，跳过已建库文件: {filename}")
                else:
                    # 只有库里没有的数据，才会执行耗时的向量提取并入库
                    vector = self.extract_image_feature(img_path)
                    self.collection.add(
                        ids=[filename],
                        embeddings=[vector],
                        metadatas=[{
                            "uid": uid,
                            "projection": str(row.get('projection', 'Unknown')),
                            "report": report_text,
                            "image_path": img_path
                        }]
                    )
                    inserted_count += 1

                    if log_callback:
                        log_callback(f"✅ 成功提取并入库: {filename} (UID: {uid})")

            # 触发前端更新进度条
            if progress_callback:
                progress_callback(min(1.0, (index + 1) / total_samples))

        if log_callback:
            log_callback(f"🏁 批次处理完成。新增: {inserted_count} 条，跳过: {skipped_count} 条。")

        return inserted_count

    def retrieve_similar_cases(self, new_img_path, top_k=3):
        """检索最相似的历史病例"""
        query_vector = self.extract_image_feature(new_img_path)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        return results

    def retrieve_weighted_fusion(self, global_img_path, roi_img_path=None, top_k=5, alpha=0.7):
        """
        加权融合检索：结合全局特征和ROI特征
        :param global_img_path: 全局图像路径
        :param roi_img_path: ROI图像路径（若为None则退化为全局检索）
        :param top_k: 返回结果数量
        :param alpha: ROI特征的权重（0~1），越大越侧重ROI区域
        :return: 与 collection.query 返回格式相同的字典
        """
        # 提取全局特征
        global_vector = np.array(self.extract_image_feature(global_img_path))

        if roi_img_path is not None and os.path.exists(roi_img_path):
            # 提取ROI特征
            roi_vector = np.array(self.extract_image_feature(roi_img_path))

            # 加权融合
            fused_vector = alpha * roi_vector + (1 - alpha) * global_vector
            # 重新归一化（保证为单位向量，使余弦距离计算正确）
            fused_vector = fused_vector / np.linalg.norm(fused_vector)
            query_vector = fused_vector.tolist()
        else:
            query_vector = global_vector.tolist()

        # 执行检索
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        return results


    def get_real_heatmap_matrix(self, query_img_path, hist_img_path):
        """
        真实特征溯源：计算 Query 图像全局特征与历史图像局部 Patch 之间的相似度矩阵
        """
        import torch
        import numpy as np
        from PIL import Image

        device = next(self.clip_model.parameters()).device

        # 1. 提取当前上传图像 (Query) 的全局向量
        query_image = self.preprocess(Image.open(query_img_path).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():
            query_feature = self.clip_model.encode_image(query_image)
            query_feature /= query_feature.norm(dim=-1, keepdim=True)
            target_dim = query_feature.shape[-1]  # 动态获取目标维度 (BiomedCLIP 中通常是 512)

        # 2. 读取历史匹配图像，准备提取它的局部特征
        hist_image = self.preprocess(Image.open(hist_img_path).convert('RGB')).unsqueeze(0).to(device)

        # 3. 注册 Hook，拦截模型池化前的输出 (获取所有 Tokens)
        activation = {}

        def hook(module, input, output):
            activation['tokens'] = output

        # 兼容 TimmModel (BiomedCLIP) 和标准 CLIP 架构
        if hasattr(self.clip_model.visual, 'trunk') and hasattr(self.clip_model.visual.trunk, 'norm'):
            target_layer = self.clip_model.visual.trunk.norm
        elif hasattr(self.clip_model.visual, 'ln_post'):
            target_layer = self.clip_model.visual.ln_post
        else:
            raise AttributeError("无法定位模型的特征输出层，请确认架构。")

        handle = target_layer.register_forward_hook(hook)

        with torch.no_grad():
            self.clip_model.encode_image(hist_image)

        handle.remove()  # 提取完毕，立刻卸载 Hook

        # 4. 获取未池化的 Patch Tokens (如: 197 x 768)
        tokens = activation['tokens']

        # 5. 特征投影降维对齐 (将 768 维平滑降到 512 维)
        projected_tokens = tokens

        # 尝试路线 A：TimmModel 架构通过 head 模块投影 (移除 isinstance 限制)
        if hasattr(self.clip_model.visual, 'head') and callable(self.clip_model.visual.head):
            try:
                projected_tokens = self.clip_model.visual.head(tokens)
            except Exception:
                pass

        # 尝试路线 B：标准 CLIP 通过 proj 矩阵投影
        if projected_tokens.shape[-1] != target_dim and hasattr(self.clip_model.visual,
                                                                'proj') and self.clip_model.visual.proj is not None:
            projected_tokens = tokens @ self.clip_model.visual.proj

        # 🛠️ 尝试路线 C (安全兜底)：如果在上述操作后，维度仍未对齐，在模型中强行搜索匹配维度的线性投影矩阵
        if projected_tokens.shape[-1] != target_dim:
            for param in self.clip_model.visual.parameters():
                if len(param.shape) == 2:
                    # 如果匹配到 (512, 768) 的权重矩阵
                    if param.shape[1] == projected_tokens.shape[-1] and param.shape[0] == target_dim:
                        projected_tokens = torch.nn.functional.linear(projected_tokens, param)
                        break
                    # 如果匹配到 (768, 512) 的权重矩阵
                    elif param.shape[0] == projected_tokens.shape[-1] and param.shape[1] == target_dim:
                        projected_tokens = torch.matmul(projected_tokens, param)
                        break

        # 6. 剥离第 0 个全局 CLS Token，保留剩下的 196 个局部 Patch 向量
        patch_tokens = projected_tokens[0, 1:, :]  # 最终形状应对齐为: (196, 512)
        patch_tokens /= patch_tokens.norm(dim=-1, keepdim=True)

        # 7. 计算历史图的各个局部区域与当前 Query 的空间相似度
        query_vector = query_feature[0]  # (512,)
        similarity = torch.matmul(patch_tokens, query_vector)  # (196,) 将不再报错！

        # 8. 重塑为 14x14 矩阵 (ViT-Base 的标准网格) 并归一化为热力图
        grid_size = int(np.sqrt(similarity.shape[0]))
        heat_matrix = similarity.view(grid_size, grid_size).detach().cpu().numpy()

        # ReLU 机制：过滤负相关区域，只关注正向相关的病灶，映射到 0~1
        heat_matrix = np.maximum(heat_matrix, 0)
        if np.max(heat_matrix) > 0:
            heat_matrix = heat_matrix / np.max(heat_matrix)

        return heat_matrix