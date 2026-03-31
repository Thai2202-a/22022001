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
# CẤU HÌNH HỆ THỐNG
# =========================
st.set_page_config(
    page_title="Đình Thái - SRT Dark Studio V12",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

BASE_SYSTEM_PROMPT = """
Bạn là chuyên gia dịch phụ đề phim chuyên nghiệp.
YÊU CẦU:
- Dịch tự nhiên, mượt, đúng ngữ cảnh hội thoại.
- Giữ văn phong phụ đề, ngắn gọn, dễ đọc.
- Không giải thích, không ghi chú, không bỏ dòng.
- Mỗi mục đầu vào phải trả về đúng 1 dòng đầu ra [số] bản_dịch.
""".strip()

SOURCE_LANGUAGE_OPTIONS = ["Tự động", "Tiếng Trung", "Tiếng Anh", "Tiếng Nhật", "Tiếng Hàn", "Tiếng Thái", "Tiếng Pháp", "Tiếng Đức"]
TARGET_LANGUAGE_OPTIONS = ["Tiếng Việt", "Tiếng Anh", "Tiếng Trung (Giản)", "Tiếng Nhật", "Tiếng Hàn", "Giống nguồn (Sửa lỗi)"]

# =========================
# GIAO DIỆN DARK MODE (CSS)
# =========================
CUSTOM_CSS = """
<style>
    .block-container {max-width: 1400px; padding-top: 1rem;}
    [data-testid="stAppViewContainer"] {background-color: #0f172a; color: #f8fafc;}
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    
    /* Thanh tiêu đề */
    .topbar {
        display:flex; justify-content:space-between; align-items:center;
        background: #1e293b; border:1px solid #334155;
        border-radius:20px; padding:20px; margin-bottom:25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    .brand-title {font-size:2rem; font-weight:900; color:#3b82f6; margin:0;}
    .brand-sub {color:#94a3b8; font-size:0.9rem; text-transform:uppercase; letter-spacing:2px;}
    
    /* Thẻ nội dung */
    .card {
        background:#1e293b; border:1px solid #334155;
        border-radius:20px; padding:20px; margin-bottom:20px;
    }
    .card-title {color:#60a5fa; font-size:1.1rem; font-weight:800; margin-bottom:15px; display:flex; align-items:center; gap:10px;}
    
    /* Input & Selectbox */
    .stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
        background-color: #0f172a !important; color: #ffffff !important; 
        border: 1px solid #475569 !important; border-radius: 12px !important;
    }
    
    /* Button */
    .stButton > button {
        border-radius:12px !important; font-weight:800 !important; transition: all 0.3s;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        border: none !important; height: 50px;
    }
    .stDownloadButton > button {
        background: #10b981 !important; color: white !important; border:none !important; border-radius:12px !important;
    }

    /* Live Box */
    .batch-live-box { background:#020617; border:1px solid #1e293b; border-radius:15px; padding:15px; min-height:200px; max-height:400px; overflow-y:auto; }
    .batch-line-pending { padding:8px; border-radius:8px; background:#1e3a8a; color:#bfdbfe; margin-bottom:8px; border-left:4px solid #3b82f6; font-size:0.9rem;}
    .batch-line-done { padding:8px; border-radius:8px; background:#064e3b; color:#a7f3d0; margin-bottom:8px; border-left:4px solid #10b981; font-size:0.9rem;}
    
    /* Metrics */
    .metric-card {background:#0f172a; border:1px solid #334155; border-radius:15px; padding:15px; text-align:center;}
    .metric-val {font-size:1.5rem; font-weight:900; color:#ffffff;}
    .metric-lab {color:#94a3b8; font-size:0.8rem;}
</style>
"""

@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""

# =========================
# LOGIC XỬ LÝ SRT & DỊCH
# =========================
def read_srt(content: str) -> List[SubtitleItem]:
    content = content.strip()
    if not content: return []
    blocks = re.split(r"\n\s*\n", content)
    items = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3: continue
        items.append(SubtitleItem(index=lines[0].strip(), timecode=lines[1].strip(), text="\n".join(lines[2:]).strip()))
    return items

def write_srt(items: List[SubtitleItem]) -> str:
    out = io.StringIO()
    for item in items:
        txt = item.translated_text if item.translated_text else item.text
        out.write(f"{item.index}\n{item.timecode}\n{txt}\n\n")
    return out.getvalue()

def build_prompt(batch: List[SubtitleItem], style: str, src: str, tar: str) -> str:
    rows = [f"[{i+1}] {item.text.replace(chr(13), '').strip()}" for i, item in enumerate(batch)]
    final_tar = tar if tar != "Giống nguồn (Sửa lỗi)" else "ngôn ngữ gốc (chỉ sửa lỗi và tối ưu)"
    return f"{BASE_SYSTEM_PROMPT}\nPHONG CÁCH: {style}\nNGUỒN: {src}\nĐÍCH: {final_tar}\n\nDANH SÁCH:\n" + "\n".join(rows)

def translate_batch(api_key, model, batch, style, src, tar):
    client = genai.Client(api_key=api_key.strip())
    prompt = build_prompt(batch, style, src, tar)
    for _ in range(MAX_RETRIES_PER_KEY):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(temperature=0.2))
            mapping = {}
            for line in resp.text.splitlines():
                match = re.match(r"^\[(\d+)\]\s*(.*)$", line.strip())
                if match: mapping[int(match.group(1))] = match.group(2).strip()
            return [mapping.get(i+1, batch[i].text) for i in range(len(batch))], True
        except: time.sleep(RETRY_SLEEP_SECONDS)
    return [item.text for item in batch], False

# =========================
# GIAO DIỆN CHÍNH
# =========================
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Topbar
st.markdown(f'''
<div class="topbar">
    <div>
        <h1 class="brand-title">ĐÌNH THÁI 🎬</h1>
        <p class="brand-sub">SRT Studio Dark V12 • {DEFAULT_MODEL}</p>
    </div>
    <div style="text-align:right">
        <div class="version-pill">FAST MODE ACTIVE</div>
    </div>
</div>
''', unsafe_allow_html=True)

col_input, col_status = st.columns([1.1, 1.5], gap="large")

with col_input:
    st.markdown('<div class="card"><div class="card-title">📁 TẢI FILE & API</div>', unsafe_allow_html=True)
    files = st.file_uploader("Chọn file SRT", type=["srt"], accept_multiple_files=True)
    api_keys = st.text_area("Danh sách API Keys (Mỗi dòng 1 key)", height=120, placeholder="AIza...")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">⚙️ CẤU HÌNH DỊCH</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("MODEL AI", [DEFAULT_MODEL, "gemini-2.0-flash", "gemini-1.5-flash"])
    
    c1, c2 = st.columns(2)
    with c1: src_lang = st.selectbox("NGÔN NGỮ NGUỒN", SOURCE_LANGUAGE_OPTIONS)
    with c2: tar_lang = st.selectbox("NGÔN NGỮ ĐÍCH", TARGET_LANGUAGE_OPTIONS)
    
    batch_val = st.number_input("BATCH SIZE", value=DEFAULT_BATCH_SIZE)
    style_val = st.text_area("PHONG CÁCH", value="Dịch tự nhiên, mượt như phim, xưng hô phù hợp ngữ cảnh.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_status:
    st.markdown('<div class="card"><div class="card-title">🚀 ĐIỀU KHIỂN</div>', unsafe_allow_html=True)
    run_btn = st.button("▶ BẮT ĐẦU XỬ LÝ", type="primary", use_container_width=True)
    
    # Khu vực hiển thị Metrics
    m1, m2, m3 = st.columns(3)
    files_count = len(files) if files else 0
    m1.markdown(f'<div class="metric-card"><div class="metric-val">{files_count}</div><div class="metric-lab">File</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-val">{tar_lang}</div><div class="metric-lab">Đích</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-val">Dark</div><div class="metric-lab">Giao diện</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📡 TIẾN TRÌNH TRỰC TIẾP</div>', unsafe_allow_html=True)
    live_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# THỰC THI DỊCH
# =========================
if run_btn:
    if not files or not api_keys:
        st.error("❌ Vui lòng chọn file và nhập API Key!")
    else:
        keys = [k.strip() for k in api_keys.splitlines() if k.strip()]
        zip_buffer = io.BytesIO()
        all_logs = []
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for f_idx, f in enumerate(files):
                content = f.read().decode("utf-8-sig", errors="ignore")
                items = read_srt(content)
                batches = [items[i:i + batch_val] for i in range(0, len(items), batch_val)]
                
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
                    future_to_batch = {
                        executor.submit(translate_batch, keys[i % len(keys)], model_choice, batches[i], style_val, src_lang, tar_lang): i 
                        for i in range(len(batches))
                    }
                    
                    for future in as_completed(future_to_batch):
                        b_id = future_to_batch[future]
                        res_texts, ok = future.result()
                        for i, txt in enumerate(res_texts):
                            batches[b_id][i].translated_text = txt
                        
                        # Cập nhật Live UI
                        log_type = "done" if ok else "pending"
                        msg = f"✓ File {f_idx+1}: Batch {b_id+1}/{len(batches)} xong."
                        all_logs.append(f'<div class="batch-line-{log_type}">{msg}</div>')
                        live_placeholder.markdown(f'<div class="batch-live-box">{"".join(all_logs[-6:])}</div>', unsafe_allow_html=True)
                
                # Ghi vào ZIP
                zf.writestr(f.name, write_srt(items).encode("utf-8"))
        
        st.success("✅ ĐÃ HOÀN THÀNH TẤT CẢ!")
        st.download_button("📥 TẢI XUỐNG KẾT QUẢ (ZIP)", zip_buffer.getvalue(), "KetQua_Dich_SRT.zip", "application/zip", use_container_width=True)

st.markdown("---")
st.caption("Developed by Đình Thái • Optimized for Gemini 2.5 Flash")
