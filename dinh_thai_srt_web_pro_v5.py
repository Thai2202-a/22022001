import io
import re
import time
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google import genai
from google.genai import types


# =========================
# CẤU HÌNH
# =========================
st.set_page_config(
    page_title="Đình Thái - SRT Translator Studio Dark",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CẬP NHẬT: Mặc định Gemini 2.5 Flash
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

BASE_SYSTEM_PROMPT = """
Bạn là chuyên gia dịch phụ đề phim chuyên nghiệp.
YÊU CẦU CHUNG:
- Dịch tự nhiên, mượt, đúng ngữ cảnh hội thoại.
- Giữ văn phong giống phụ đề phim, dễ đọc, gọn.
- Không giải thích, không ghi chú, không bỏ dòng, không đánh số lại.
- Mỗi mục phụ đề đầu vào phải trả về đúng 1 dòng đầu ra tương ứng.
""".strip()

SOURCE_LANGUAGE_OPTIONS = ["Tự động", "Tiếng Trung", "Tiếng Anh", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Thái", "Tiếng Pháp", "Tiếng Đức", "Tiếng Nga"]

# THÊM: Danh sách ngôn ngữ đích
TARGET_LANGUAGE_OPTIONS = ["Tiếng Việt", "Tiếng Anh", "Tiếng Trung (Giản)", "Tiếng Nhật", "Tiếng Hàn", "Giống nguồn (Sửa lỗi)"]

LANGUAGE_LABELS = {"zh": "Tiếng Trung", "ja": "Tiếng Nhật", "ko": "Tiếng Hàn", "th": "Tiếng Thái", "vi": "Tiếng Việt", "en_or_latin": "Tiếng Anh", "unknown": "Không xác định"}

# CẬP NHẬT: CUSTOM CSS - NỀN ĐEN CHỮ TRẮNG
CUSTOM_CSS = """
<style>
.block-container {max-width: 1450px; padding-top: .8rem; padding-bottom: 1.2rem;}
html, body, [class*="css"] {font-family: Inter, sans-serif; background-color: #0f172a;}
[data-testid="stAppViewContainer"] {background: #0f172a; color: #ffffff;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}

.topbar {
    display:flex; justify-content:space-between; align-items:center;
    background: #1e293b; border:1px solid #334155; box-shadow:0 10px 30px rgba(0,0,0,0.4);
    border-radius:24px; padding:16px 18px; margin-bottom:18px;
}
.brand-title {font-size:1.95rem; font-weight:900; color:#3b82f6;}
.brand-sub {color:#94a3b8; letter-spacing:.16em; text-transform:uppercase; font-size:.82rem;}

.card {
    background:#1e293b; border:1px solid #334155;
    border-radius:24px; padding:18px; margin-bottom:18px;
    box-shadow:0 10px 30px rgba(0,0,0,0.3); color: #ffffff;
}
.card-title {color:#60a5fa; font-size:1.06rem; font-weight:800; margin-bottom:12px;}

.metric-card {background:#0f172a; border:1px solid #334155; border-radius:18px; padding:14px; text-align: center;}
.metric-value {color:#ffffff; font-size:1.42rem; font-weight:900;}

/* Input Dark Mode */
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div {
    background-color: #0f172a !important; color: #ffffff !important; border: 1px solid #475569 !important; border-radius: 12px !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
    color: white !important; border: none !important; box-shadow: 0 4px 15px rgba(37,99,235,0.4);
}

.batch-live-box { background:#000000; border:1px solid #334155; border-radius:18px; padding:14px; min-height:220px; }
.batch-line-pending { background:#1e3a8a; color:#bfdbfe; padding:8px; border-radius:8px; margin-bottom:8px; border-left:4px solid #2563eb; }
.batch-line-done { background:#064e3b; color:#a7f3d0; padding:8px; border-radius:8px; margin-bottom:8px; border-left:4px solid #10b981; }

code { color: #f8fafc !important; background: #000000 !important; }
</style>
"""

@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""

# =========================
# XỬ LÝ SRT & DỊCH (UPDATE)
# =========================
def read_srt_content(content: str) -> List[SubtitleItem]:
    content = content.strip()
    if not content: return []
    blocks = re.split(r"\n\s*\n", content)
    items = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3: continue
        items.append(SubtitleItem(index=lines[0].strip(), timecode=lines[1].strip(), text="\n".join(lines[2:]).strip()))
    return items

def write_srt_content(items: List[SubtitleItem]) -> str:
    output = io.StringIO()
    for item in items:
        text = item.translated_text.strip() if item.translated_text.strip() else item.text.strip()
        output.write(f"{item.index}\n{item.timecode}\n{text}\n\n")
    return output.getvalue()

def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str, target_language: str) -> str:
    rows = [f"[{i+1}] {item.text.replace(chr(13), '').strip()}" for i, item in enumerate(batch)]
    joined_rows = "\n".join(rows)
    
    # Logic xác định ngôn ngữ đích
    final_target = target_language if target_language != "Giống nguồn (Sửa lỗi)" else "ngôn ngữ gốc"
    
    prompt = (
        f"{BASE_SYSTEM_PROMPT}\n"
        f"PHONG CÁCH: {style_prompt}\n"
        f"NHIỆM VỤ: Dịch tất cả sang **{final_target}**.\n"
        f"NGÔN NGỮ NGUỒN: {source_language}\n"
        "ĐỊNH DẠNG TRẢ VỀ: [số] bản_dịch\n\n"
        f"DANH SÁCH:\n{joined_rows}"
    )
    return prompt.strip()

# Các hàm core giữ nguyên logic từ file gốc của bạn nhưng thêm tham số target_language
def try_translate_batch_with_key(api_key, model_name, batch, style, src, tar):
    client = genai.Client(api_key=api_key.strip())
    prompt = build_prompt(batch, style, src, tar)
    for _ in range(MAX_RETRIES_PER_KEY):
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (response.text or "").strip()
            # Parsing đơn giản
            mapping = {}
            for line in text.splitlines():
                match = re.match(r"^\[(\d+)\]\s*(.*)$", line.strip())
                if match: mapping[int(match.group(1))] = match.group(2).strip()
            return [mapping.get(i+1, batch[i].text) for i in range(len(batch))]
        except: time.sleep(RETRY_SLEEP_SECONDS)
    return [item.text for item in batch]

def translate_batch_with_failover(batch_id, batch, worker_slots, model_name, style, src, tar):
    for api_key in worker_slots:
        try:
            translated = try_translate_batch_with_key(api_key, model_name, batch, style, src, tar)
            return batch_id, True, translated, ""
        except Exception as e: last_error = str(e)
    return batch_id, False, [item.text for item in batch], last_error

def process_one_file(file_name, source_bytes, partial_bytes, worker_slots, model_name, style, batch_size, src_lang, tar_lang, progress_callback=None):
    source_text = source_bytes.decode("utf-8-sig", errors="ignore")
    source_items = read_srt_content(source_text)
    if not source_items: return {"file_name": file_name, "success": False, "output_bytes": b"", "stats": {"total": 0, "done": 0}, "logs": ["Lỗi file"]}
    
    batches = [source_items[i:i + batch_size] for i in range(0, len(source_items), batch_size)]
    done_lines = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        futures = [executor.submit(translate_batch_with_failover, i, b, worker_slots, model_name, style, src_lang, tar_lang) for i, b in enumerate(batches)]
        for future in as_completed(futures):
            b_id, ok, trans, err = future.result()
            for i, txt in enumerate(trans): batches[b_id][i].translated_text = txt
            done_lines += len(trans)
            if progress_callback: progress_callback("batch_done", file_name, b_id, len(batches), trans[:1])

    return {
        "file_name": file_name, "success": True, "output_bytes": write_srt_content(source_items).encode("utf-8"),
        "stats": {"total": len(source_items), "done": done_lines, "failed_batches": 0, "speed": done_lines/max(0.001, time.time()-start_time)},
        "logs": [f"Xong {file_name}"], "detected_lang": "OK"
    }

# =========================
# UI RENDER
# =========================
init_state = lambda: [st.session_state.setdefault(k, v) for k, v in {"run_logs": [], "stats": {"files":0,"total":0,"done":0,"failed_batches":0}, "speed_text": "0 dòng/s", "live_pending_lines":[], "live_done_lines":[]}.items()]
init_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="topbar"><div class="brand-wrap"><div class="brand-title">Đình Thái 🎬</div><div class="brand-sub">SRT DARK STUDIO</div></div><div class="version-pill">GEMINI 2.5 FLASH</div></div>', unsafe_allow_html=True)

left, right = st.columns([1.1, 1.5], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-title">📤 Upload Files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Chọn file SRT", type=["srt"], accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">⚙️ Cấu Hình</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: model_name = st.selectbox("MODEL", [DEFAULT_MODEL, "gemini-2.0-flash"])
    with c2: src_lang = st.selectbox("NGUỒN", SOURCE_LANGUAGE_OPTIONS)
    with c3: tar_lang = st.selectbox("ĐÍCH", TARGET_LANGUAGE_OPTIONS) # NÚT CHỌN NGÔN NGỮ ĐÍCH
    
    batch_size = st.number_input("BATCH SIZE", value=80)
    style_prompt = st.text_area("Prompt", value="Dịch tự nhiên, mượt như phụ đề phim.")
    keys_text = st.text_area("API Keys", height=100)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card"><div class="card-title">🚀 Điều Khiển</div>', unsafe_allow_html=True)
    run_btn = st.button("▶ BẮT ĐẦU DỊCH", type="primary", use_container_width=True)
    
    # Metrics hiển thị
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div class="metric-card"><div class="metric-value">{len(uploaded_files) if uploaded_files else 0}</div><div class="small">File</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state["speed_text"]}</div><div class="small">Tốc độ</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-value">{tar_lang}</div><div class="small">Đích</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📡 Tiến Trình Live</div>', unsafe_allow_html=True)
    live_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# THỰC THI
if run_btn:
    if not uploaded_files or not keys_text: st.error("Thiếu File hoặc Key!")
    else:
        worker_slots = [k.strip() for k in keys_text.splitlines() if k.strip()]
        zip_buffer = io.BytesIO()
        start_time_all = time.time()
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for idx, f in enumerate(uploaded_files):
                def ui_cb(event, fname, bid, btotal, blines):
                    msg = f"✓ {fname} | Batch {bid+1}/{btotal}"
                    st.session_state["live_done_lines"].append(msg)
                    live_placeholder.markdown(f'<div class="batch-live-box">{"".join([f"<div class='batch-line-done'>{x}</div>" for x in st.session_state["live_done_lines"][-8:]])}</div>', unsafe_allow_html=True)

                res = process_one_file(f.name, f.read(), None, worker_slots, model_name, style_prompt, batch_size, src_lang, tar_lang, ui_cb)
                zf.writestr(res["file_name"], res["output_bytes"])
                st.session_state["speed_text"] = f"{res['stats']['speed']:.1f} d/s"

        # ÂM THANH THÔNG BÁO KHI XONG
        st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/button-30.mp3" type="audio/mpeg"></audio>""", height=0)
        
        st.success("ĐÃ XONG!")
        st.download_button("📥 Tải ZIP kết quả", zip_buffer.getvalue(), "srt_translated.zip", use_container_width=True)

st.caption("Developed by Đình Thái • Optimized for Dark Mode")
