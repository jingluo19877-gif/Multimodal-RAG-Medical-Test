import streamlit as st
import os
from rag_engine import MedicalRAGEngine
from llm_service import MedicalLLMService

st.set_page_config(page_title="医学影像 RAG", layout="wide")


# --- 1. 初始化后端服务 (使用 st.cache_resource 避免每次刷新网页都重新加载模型) ---
@st.cache_resource
def load_services():
    # 填入你 Colab 的 ngrok 临时地址
    llm = MedicalLLMService(remote_url="https://mistakenly-hungry-tonie.ngrok-free.dev")
    rag = MedicalRAGEngine()
    return llm, rag


llm_service, rag_engine = load_services()

# --- 2. 绘制 UI 界面 ---
st.title("🩺 多模态医学影像 RAG 诊断系统")

tab_admin, tab_chat = st.tabs(["⚙️ 数据集处理与建库", "💬 多模态问诊"])

with tab_admin:
    st.header("构建多模态索引库")

    # 根据你提供的真实绝对路径设置默认值
    default_base = r"E:\Work SoftWare\physicalTest\data\iu_xray\datasets\raddar\chest-xrays-indiana-university\versions\2"
    base_dir = st.text_input("数据集根目录", value=default_base)

    if st.button("🚀 开始解析并入库"):
        with st.spinner("正在联合解析 reports 和 projections，并提取向量..."):
            reports_csv = os.path.join(base_dir, "indiana_reports.csv")
            projections_csv = os.path.join(base_dir, "indiana_projections.csv")
            img_dir = os.path.join(base_dir, "images", "images_normalized")

            if not os.path.exists(reports_csv) or not os.path.exists(img_dir):
                st.error("找不到指定的 CSV 或图片目录，请检查路径！")
            else:
                # 为了测试，我们先跑 20 条数据
                count = rag_engine.build_index_from_csv(reports_csv, projections_csv, img_dir, max_samples=20)
                st.success(f"建库成功！共处理 {count} 份多模态影像。")

with tab_chat:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("1. 上传新患者影像")
        uploaded_file = st.file_uploader("请选择一张 X-Ray 图像", type=["png", "jpg"])
        if uploaded_file:
            # 将上传的图片临时保存，供后端提取特征和发送给大模型
            temp_img_path = f"temp_{uploaded_file.name}"
            with open(temp_img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(temp_img_path, caption="当前患者影像", use_container_width=True)

    with col2:
        st.header("2. AI 辅助分析")
        user_query = st.text_input("输入指令:", value="请详细分析这张胸片中的异常区域。")

        if st.button("开始 RAG 诊断") and uploaded_file:
            # 步骤 A: 检索相似病例
            with st.spinner("正在向量库中检索相似病例..."):
                results = rag_engine.retrieve_similar_cases(temp_img_path, top_k=2)

                # 整理检索出的历史报告作为 Context
                context_reports = ""
                if results and results['metadatas'][0]:
                    st.write("🔍 **参考的历史英文病历:**")
                    for i, meta in enumerate(results['metadatas'][0]):
                        st.info(f"病例 {i + 1}: {meta['report']}")
                        context_reports += f"Case {i + 1}: {meta['report']}\n"

            # 步骤 B: 流式调用大模型进行诊断
            st.write("🧠 **AI 综合诊断建议:**")
            with st.spinner("MedGemma 正在思考..."):
                # st.write_stream 完美接收 yield 生成器，实现打字机效果！
                stream_generator = llm_service.stream_diagnosis(temp_img_path, user_query, context_reports)
                st.write_stream(stream_generator)