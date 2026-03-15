import streamlit as st
import os
import json
from datetime import datetime
import uuid

from rag_engine import MedicalRAGEngine
from llm_service import MedicalLLMService

from streamlit_drawable_canvas import st_canvas
from PIL import Image
from image_utils import generate_overlay_heatmap
import numpy as np
import cv2

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="医疗影像辅助诊断系统", page_icon="🩻", layout="wide")

# ==========================================
# 加载后端服务（缓存）
# ==========================================
@st.cache_resource
def load_backend_services():
    REMOTE_LLM_URL = "https://mistakenly-hungry-tonie.ngrok-free.dev/"  # 请替换为实际地址
    with st.spinner("正在初始化 BiomedCLIP 引擎与大模型连接..."):
        rag = MedicalRAGEngine(db_path="./storage/medical_vectordb")
        llm = MedicalLLMService(remote_url=REMOTE_LLM_URL)
    return rag, llm

rag_engine, llm_service = load_backend_services()

# ==========================================
# 目录初始化
# ==========================================
RAW_IMG_DIR = os.path.join(os.getcwd(), "data", "raw_images")
os.makedirs(RAW_IMG_DIR, exist_ok=True)

# ==========================================
# 会话历史持久化（JSON文件）
# ==========================================
CHAT_HISTORY_FILE = "./chat_history.json"

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_chat_history(chats):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)

# 初始化 session_state
if "chats" not in st.session_state:
    if "renaming_chat_id" not in st.session_state:
        st.session_state.renaming_chat_id = None
    st.session_state.chats = load_chat_history()
    if not st.session_state.chats:
        default_id = str(uuid.uuid4())[:8]
        st.session_state.chats = {
            default_id: {
                "title": "新会话",
                "messages": [],
                "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        save_chat_history(st.session_state.chats)
    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]

def create_new_chat():
    new_id = str(uuid.uuid4())[:8]
    st.session_state.chats[new_id] = {
        "title": f"新会话 {new_id[:4]}",
        "messages": [],
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.current_chat_id = new_id
    save_chat_history(st.session_state.chats)

def delete_chat(chat_id):
    if len(st.session_state.chats) > 1:
        del st.session_state.chats[chat_id]
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[-1]
        save_chat_history(st.session_state.chats)
        st.rerun()

def update_chat_title(chat_id, title):
    if chat_id in st.session_state.chats:
        st.session_state.chats[chat_id]["title"] = title
        st.session_state.chats[chat_id]["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_chat_history(st.session_state.chats)

def add_message_to_chat(chat_id, message):
    st.session_state.chats[chat_id]["messages"].append(message)
    st.session_state.chats[chat_id]["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_chat_history(st.session_state.chats)

# ==========================================
# 侧边栏（操作面板）
# ==========================================
with st.sidebar:
    st.title("🩻 操作面板")

    if st.button("➕ 新建问诊会话", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    st.subheader("对话历史")

    # 获取所有会话的 id 和标题
    chat_ids = list(st.session_state.chats.keys())
    chat_titles = [st.session_state.chats[chat_id]["title"] for chat_id in chat_ids]

    # 确保当前会话 id 有效
    if st.session_state.current_chat_id not in chat_ids:
        st.session_state.current_chat_id = chat_ids[0]

    # 下拉框选择当前会话
    selected_index = chat_ids.index(st.session_state.current_chat_id)
    new_index = st.selectbox(
        "选择会话",
        range(len(chat_ids)),
        format_func=lambda i: chat_titles[i],
        index=selected_index,
        key="chat_selector",
        label_visibility="collapsed"  # 隐藏标签，节省空间
    )
    if new_index != selected_index:
        st.session_state.current_chat_id = chat_ids[new_index]
        st.rerun()

    # 重命名和删除按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("重命名", use_container_width=True):
            st.session_state.renaming_chat_id = st.session_state.current_chat_id
            st.rerun()
    with col2:
        if st.button("删除", use_container_width=True):
            if len(st.session_state.chats) > 1:
                chat_id_to_delete = st.session_state.current_chat_id
                # 切换到其他会话
                new_current = next((cid for cid in chat_ids if cid != chat_id_to_delete), None)
                del st.session_state.chats[chat_id_to_delete]
                save_chat_history(st.session_state.chats)
                st.session_state.current_chat_id = new_current
                st.rerun()
            else:
                st.warning("至少保留一个会话")

    # 如果正在重命名当前会话，显示输入框
    if st.session_state.get("renaming_chat_id") == st.session_state.current_chat_id:
        current_title = st.session_state.chats[st.session_state.current_chat_id]["title"]
        new_title = st.text_input("新标题", value=current_title, key="rename_input")
        if st.button("保存", key="save_rename"):
            if new_title.strip():
                update_chat_title(st.session_state.current_chat_id, new_title.strip())
            st.session_state.renaming_chat_id = None
            st.rerun()

    st.divider()

    # ========== 引擎状态改为纵向卡片 ==========
    st.subheader("状态")

    # 知识库卡片
    with st.container(border=True):
        st.markdown("**知识库**")
        st.markdown("已连接")

    # 检索模型卡片
    with st.container(border=True):
        st.markdown("**检索模型**")
        st.markdown("BiomedCLIP")

    # 生成模型卡片
    with st.container(border=True):
        st.markdown("**生成模型**")
        st.markdown("MedGemma-4B")

    st.divider()

    # ========== 高级检索设置 ==========
    st.subheader("高级检索设置")
    top_k_slider = st.slider("召回数量 (Top-K)", min_value=1, max_value=5, value=2)
    st.subheader("融合检索权重")
    alpha_slider = st.slider(
        "ROI 权重 (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="权重越大，检索结果越侧重用户圈选的ROI区域"
    )


st.title("医疗影像辅助诊断系统")
# ==========================================
# 主内容区标签页
# ==========================================
tab_kb, tab_chat, tab_monitor = st.tabs(["知识库", "💬 影像问答", "系统状态"])

# ---------------------------------------------------------
# Tab 1: 知识库构建
# ---------------------------------------------------------
with tab_kb:
    st.header("知识库构建")
    st.caption("从 IU X-Ray 数据集中提取影像与报告，建立多模态索引。")

    default_base = r"E:\Work SoftWare\physicalTest\data\iu_xray\datasets\raddar\chest-xrays-indiana-university\versions\2"
    base_dir = st.text_input("数据集根目录", value=default_base)
    build_limit = st.number_input("样本数量（测试建议 20-50）", min_value=1, value=20)

    if st.button("开始构建索引", type="primary"):
        reports_csv = os.path.join(base_dir, "indiana_reports.csv")
        projections_csv = os.path.join(base_dir, "indiana_projections.csv")
        img_dir = os.path.join(base_dir, "images", "images_normalized")

        if not os.path.exists(reports_csv) or not os.path.exists(img_dir):
            st.error("找不到指定的 CSV 或图片目录，请检查路径！")
        else:
            st.write("### 构建进度")
            progress_bar = st.progress(0)
            log_box = st.empty()

            def update_progress(p):
                progress_bar.progress(p)

            def update_log(msg):
                log_box.info(msg)

            with st.spinner("正在提取特征并建立索引..."):
                count = rag_engine.build_index_from_csv(
                    reports_csv, projections_csv, img_dir,
                    max_samples=build_limit,
                    progress_callback=update_progress,
                    log_callback=update_log
                )
                st.success(f"✅ 索引构建完成！成功入库 {count} 张图像及报告。")

# ---------------------------------------------------------
# Tab 2: 影像问答
# ---------------------------------------------------------
with tab_chat:
    st.header("影像辅助诊断")
    current_chat = st.session_state.chats[st.session_state.current_chat_id]

    # ==========================================
    # 第一部分：渲染历史消息 (修复区：这里只读取，不运算)
    # ==========================================
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "images" in msg and msg["images"]:
                cols = st.columns(len(msg["images"]))
                for idx, path in enumerate(msg["images"]):
                    cols[idx].image(path, width=200)

            # 渲染检索到的病例证据 (这里不再运算，直接从历史数据里拿)
            if "retrieved_cases" in msg and msg["retrieved_cases"]:
                st.markdown("##### 🔍 相似历史病例")
                for i, case in enumerate(msg["retrieved_cases"]):
                    with st.container(border=True):
                        st.markdown(f"**病例 {i + 1}** (相似度: `{case['dist']:.3f}` | UID: `{case['uid']}`)")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if os.path.exists(case['image_path']):
                                st.image(case['image_path'], use_container_width=True)
                            else:
                                st.error(f"图片不存在：{case['image_path']}")
                        with col2:
                            st.caption("原始报告 (英文):")
                            st.info(case['report'])

            # 渲染RAG执行详情
            if "step1_log" in msg and "step2_log" in msg and "step3_log" in msg:
                st.markdown("**🔍 RAG执行详情**")
                with st.container(border=True):
                    st.markdown("**步骤1：查询翻译**")
                    st.text(msg["step1_log"])
                with st.container(border=True):
                    st.markdown("**步骤2：相似病例检索**")
                    st.text(msg["step2_log"])
                with st.container(border=True):
                    st.markdown("**步骤3：生成诊断建议**")
                    st.text(msg["step3_log"])
            elif "rag_log" in msg and msg["rag_log"]:
                st.markdown("**🔍 RAG执行详情**")
                st.text_area("RAG执行日志", value=msg["rag_log"], height=150, disabled=True,
                             label_visibility="collapsed")

    # ==========================================
    # 第二部分：底部输入与 ROI 画板区
    # ==========================================
    with st.container(border=True):
        st.caption("📎 上传本次问诊影像（支持多张，将使用第一张进行检索）")
        chat_images = st.file_uploader(
            "上传影像", type=["png", "jpg", "jpeg", "dcm"],
            accept_multiple_files=True, label_visibility="collapsed"
        )

        roi_img_path = None
        if chat_images:
            st.caption("🖌️ 可选操作：请在下方影像上框选病灶区域 (ROI)，我们将针对该区域进行精准检索。")
            bg_image = Image.open(chat_images[0])
            canvas_width = 400
            canvas_height = int(canvas_width * (bg_image.height / bg_image.width))

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=bg_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key="roi_canvas",
            )

    prompt = st.chat_input("请输入您的问题（例如：请分析图中病灶）...")

    # ==========================================
    # 第三部分：触发 RAG 实时检索 (热力图计算区)
    # ==========================================
    if prompt:
        saved_img_paths = []
        if chat_images:
            session_img_dir = os.path.join(RAW_IMG_DIR, st.session_state.current_chat_id)
            os.makedirs(session_img_dir, exist_ok=True)

            for img in chat_images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(session_img_dir, f"query_raw_{timestamp}.{img.name.split('.')[-1]}")
                with open(save_path, "wb") as f:
                    f.write(img.getbuffer())
                saved_img_paths.append(save_path)

            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                last_rect = canvas_result.json_data["objects"][-1]
                scale_x = bg_image.width / canvas_width
                scale_y = bg_image.height / canvas_height

                left = int(last_rect["left"] * scale_x)
                top = int(last_rect["top"] * scale_y)
                right = int((last_rect["left"] + last_rect["width"]) * scale_x)
                bottom = int((last_rect["top"] + last_rect["height"]) * scale_y)

                roi_img = bg_image.crop((left, top, right, bottom))
                roi_save_path = os.path.join(session_img_dir, f"query_roi_{timestamp}.png")
                roi_img.save(roi_save_path)
                roi_img_path = roi_save_path
                st.info("🎯 检测到 ROI 选区，已启用局部特征增强检索。")

            with st.chat_message("user"):
                st.markdown(prompt)
                if saved_img_paths:
                    display_path = roi_img_path if roi_img_path else saved_img_paths[0]
                    st.image(display_path, width=200, caption="检索特征源 (Target Image)")

        user_msg = {"role": "user", "content": prompt, "images": saved_img_paths}
        add_message_to_chat(st.session_state.current_chat_id, user_msg)

        with st.chat_message("assistant"):
            if not saved_img_paths:
                st.warning("未检测到图片，请上传影像以获得精准分析。")
                final_answer = "请上传图片以便进行多模态分析。"
                st.markdown(final_answer)
                step1_log = step2_log = step3_log = "未上传图片，跳过RAG流程。"
                saved_cases = []
            else:
                #target_img = roi_img_path if roi_img_path else saved_img_paths[0]
                retrieval_img = roi_img_path if roi_img_path else saved_img_paths[0]  # 局部特征用于去 ChromaDB 找相似病例
                full_original_img = saved_img_paths[0]  # 全局原图永远留给 MedGemma 进行整体阅片

                step1_placeholder = st.empty()
                step2_placeholder = st.empty()
                step3_placeholder = st.empty()

                step1_log = step2_log = step3_log = ""

                with step1_placeholder.container():
                    st.markdown("##### 🧠 步骤1：查询翻译")
                    translated_query = llm_service.translate_to_english(prompt)
                    step1_log = f"原始输入：{prompt}\n英文意图：{translated_query}"
                    st.info(f"**原始输入**：{prompt}")
                    st.success(f"**英文意图**：{translated_query}")

                with step2_placeholder.container():
                    st.markdown("##### 🔍 步骤2：相似病例检索")
                    with st.spinner("正在检索相似病例并生成特征热力图..."):
                        #results = rag_engine.retrieve_similar_cases(target_img, top_k=top_k_slider)
                        #results = rag_engine.retrieve_similar_cases(retrieval_img, top_k=top_k_slider)
                        if roi_img_path and os.path.exists(roi_img_path):
                            # 使用加权融合检索
                            results = rag_engine.retrieve_weighted_fusion(
                                global_img_path=full_original_img,
                                roi_img_path=roi_img_path,
                                top_k=top_k_slider,
                                alpha=alpha_slider  # 从侧边栏获取用户设置
                            )
                            st.info(f"已启用加权融合检索，α = {alpha_slider:.2f}")
                        else:
                            # 无ROI时使用普通检索
                            results = rag_engine.retrieve_similar_cases(retrieval_img, top_k=top_k_slider)

                    if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
                        st.success(f"检索成功，发现 {len(results['ids'][0])} 份相似病例。")
                        context_reports = ""
                        saved_cases = []
                        step2_log = f"成功召回 {len(results['ids'][0])} 份病例：\n"

                        for i, (meta, dist) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                            hist_img_path = meta.get('image_path', '')
                            hist_report = meta.get('report', '')
                            context_reports += f"Case {i + 1} (Distance: {dist:.3f}):\n{hist_report}\n\n"

                            if not os.path.exists(hist_img_path):
                                filename = os.path.basename(hist_img_path)
                                fallback_path = os.path.join(os.getcwd(), "data", "iu_xray", "datasets", "raddar",
                                                             "chest-xrays-indiana-university", "versions", "2",
                                                             "images",
                                                             "images_normalized", filename)
                                if os.path.exists(fallback_path):
                                    hist_img_path = fallback_path

                            saved_cases.append({
                                "uid": meta.get('uid'),
                                "dist": dist,
                                "image_path": hist_img_path,
                                "report": hist_report
                            })

                            # --- 就在这里！生成并展示特征热力图 ---
                            with st.container(border=True):
                                st.markdown(f"**病例 {i + 1}** (相似度距离: `{dist:.3f}` | UID: `{meta.get('uid')}`)")
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    if os.path.exists(hist_img_path):
                                        #real_matrix = rag_engine.get_real_heatmap_matrix(target_img, hist_img_path)
                                        real_matrix = rag_engine.get_real_heatmap_matrix(retrieval_img, hist_img_path)
                                        heatmap_path = generate_overlay_heatmap(hist_img_path, real_matrix)

                                        img_tab1, img_tab2 = st.tabs(["相似区域热力图", "原始医学影像"])
                                        img_tab1.image(heatmap_path, use_container_width=True,
                                                       caption="热力图叠加")
                                        img_tab2.image(hist_img_path, use_container_width=True, caption="Raw X-Ray")
                                    else:
                                        st.error(f"图片不存在：{hist_img_path}")
                                with col2:
                                    st.caption("原始报告 (英文):")
                                    st.info(hist_report)

                            step2_log += f"- 召回病例 {i + 1}: UID {meta.get('uid')}, 距离 {dist:.3f}\n"
                    else:
                        st.warning("未检索到相似病例，将仅依据用户输入生成回答。")
                        context_reports = ""
                        saved_cases = []
                        step2_log = "知识库中无匹配数据。"

                with step3_placeholder.container():
                    st.markdown("##### 📝 步骤3：生成诊断建议")

                    # 收集英文流式输出
                    full_english = ""
                    english_placeholder = st.empty()

                    with st.container(border=True):
                        # 构造 ROI 描述
                        roi_desc = None
                        if roi_img_path:
                            roi_desc = "region circled by user"

                        stream_gen = llm_service.stream_diagnosis(
                            full_original_img,
                            translated_query,  # 使用步骤1中翻译好的英文主诉
                            context_reports,
                            roi_description=roi_desc
                        )

                        # 实时显示英文
                        for chunk in stream_gen:
                            full_english += chunk
                            english_placeholder.markdown(full_english)

                        # 翻译为中文
                        with st.spinner("正在翻译为中文..."):
                            chinese_report = llm_service.translate_to_chinese(full_english)

                        # 清空英文占位，用标签页展示中英文
                        english_placeholder.empty()
                        tab_en, tab_zh = st.tabs(["英文原文", "中文翻译"])
                        tab_en.markdown(full_english)
                        tab_zh.markdown(chinese_report)

                    step3_log = "触发 MedGemma 流式输出并完成翻译。"

            assistant_msg = {
                "role": "assistant",
                "content": final_answer if 'final_answer' in locals() else "请上传图片进行分析。",
                "step1_log": step1_log if 'step1_log' in locals() else "",
                "step2_log": step2_log if 'step2_log' in locals() else "",
                "step3_log": step3_log if 'step3_log' in locals() else "",
                "retrieved_cases": saved_cases if 'saved_cases' in locals() else []
            }
            add_message_to_chat(st.session_state.current_chat_id, assistant_msg)

            if len(current_chat["messages"]) == 2:
                new_title = prompt[:10] + "..." if len(prompt) > 10 else prompt
                update_chat_title(st.session_state.current_chat_id, new_title)

# ---------------------------------------------------------
# Tab 3: 系统状态
# ---------------------------------------------------------
with tab_monitor:
    st.header("系统运行状态")
    try:
        total_docs = rag_engine.collection.count()
    except:
        total_docs = 0
    st.metric("ChromaDB 向量总数", f"{total_docs} 条")
    st.metric("当前会话数", len(st.session_state.chats))
    st.info("提示：请先在“知识库”标签页中构建索引，再进行影像问答。")