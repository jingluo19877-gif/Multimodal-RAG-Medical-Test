import os
import torch
import chromadb
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer



print("1. 正在加载 BiomedCLIP...")
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
model.eval()

print("2. 初始化本地 ChromaDB 向量库...")
# 数据会保存在当前目录的 medical_vectordb 文件夹下
chroma_client = chromadb.PersistentClient(path="./medical_vectordb")
# 创建一个名为 "medical_images" 的集合，使用余弦相似度
collection = chroma_client.get_or_create_collection(
    name="medical_images",
    metadata={"hnsw:space": "cosine"}
)

# 假设我们在库里存入两张参考图片的特征和对应的报告
# （这里你需要换成你本地实际存在的两张测试图片路径）
reference_images = [r'D:\TestPicture\1.png', r'D:\TestPicture\2.png']
reports = [
    "患者胸部X光未见明显异常，肺纹理清晰，心影大小正常。",
    "右肺上叶可见斑片状高密度影，边缘模糊，疑似炎症或结核灶。"
]

print("3. 提取特征并存入数据库...")
for i, img_path in enumerate(reference_images):
    # 提取图片向量
    image = preprocess(Image.open(img_path)).unsqueeze(0)
    with torch.no_grad():
        img_feature = model.encode_image(image)
        img_feature /= img_feature.norm(dim=-1, keepdim=True)
        # 转成普通的 Python list
        vector = img_feature.cpu().numpy().tolist()[0]

        # 存入数据库：ID, 向量, 以及对应的医生报告作为 Metadata
    collection.add(
        ids=[f"case_{i}"],
        embeddings=[vector],
        metadatas=[{"report": reports[i], "image_path": img_path}]
    )

print("✅ 历史病历建库完成！")

# ---------------------------------------------------------
# 模拟问诊：用户上传了一张新图片
print("\n4. 模拟患者上传新图片进行检索...")
new_patient_img = r'D:\TestPicture\3.png'  # 用户传的新图
new_image = preprocess(Image.open(new_patient_img)).unsqueeze(0)

with torch.no_grad():
    new_feature = model.encode_image(new_image)
    new_feature /= new_feature.norm(dim=-1, keepdim=True)
    query_vector = new_feature.cpu().numpy().tolist()[0]

# 在库中寻找最相似的 1 个历史病例
results = collection.query(
    query_embeddings=[query_vector],
    n_results=1
)

print(f"🔍 检索到最相似的历史病例 ID: {results['ids'][0][0]}")
print(f"📄 对应的历史诊断报告: {results['metadatas'][0][0]['report']}")
print(f"📏 相似度距离 (越小越好): {results['distances'][0][0]}")