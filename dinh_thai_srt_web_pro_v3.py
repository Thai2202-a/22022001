import io
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google import genai
from google.genai import types


# =========================
# CẤU HÌNH CHUNG
# =========================
st.set_page_config(
    page_title="Đình Thái - SRT Translator Pro V3",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 30
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.6

BASE_SYSTEM_PROMPT = """
Bạn là chuyên gia dịch phụ đề phim từ tiếng Trung sang tiếng Việt.

YÊU CẦU CHUNG:
- Dịch tự nhiên, mượt, đúng ngữ cảnh hội thoại.
- Giữ văn phong giống phụ đề phim, dễ đọc, gọn.
- Không giải thích, không ghi chú.
- Không đánh số lại.
- Không bỏ dòng nào.
- Mỗi mục phụ đề đầu vào phải trả về đúng 1 dòng đầu ra tương ứng.
- Nếu gặp tên riêng thì xử lý hợp lý theo ngữ cảnh.
- Không thêm ký tự thừa.
""".strip()

CUSTOM_CSS = """
<style>

/* ===== NỀN APP ===== */
body, .stApp {
    background:#050505;
    color:#facc15;
}

/* ===== HEADER ===== */
.topbar{
    display:flex;
    justify-content:space-between;
    align-items:center;
    border-bottom:1px solid #222;
    margin-bottom:20px;
    padding-bottom:10px;
}

.brand-title{
    font-size:32px;
    font-weight:900;
    color:#facc15;
    text-shadow:0 0 6px rgba(250,204,21,0.5);
}

.brand-sub{
    font-size:12px;
    color:#aaa;
    letter-spacing:2px;
}

/* ===== CARD ===== */
.card{
    background:#0b0b0b;
    border:1px solid #1f1f1f;
    border-radius:18px;
    padding:18px;
    margin-bottom:18px;
}

/* ===== TITLE ===== */
.card-title{
    color:#facc15;
    font-weight:700;
    margin-bottom:10px;
}

/* ===== INPUT ===== */
input, textarea{
    background:#000 !important;
    color:#facc15 !important;
    border:1px solid #333 !important;
}

/* ===== FILE UPLOAD ===== */
[data-testid="stFileUploaderDropzone"]{
    background:#0a0a0a;
    border:2px dashed #333;
    border-radius:16px;
}

/* ===== BUTTON ===== */
.stButton > button{
    background:linear-gradient(90deg,#7f1d1d,#dc2626);
    color:white;
    border-radius:12px;
    border:none;
    height:50px;
    font-weight:700;
}

/* ===== PROGRESS ===== */
[data-testid="stProgressBar"] div div div{
    background:#dc2626;
}

/* ===== CONSOLE ===== */
.console{
    background:#000;
    border:1px solid #333;
    border-radius:14px;
}

/* ===== PREVIEW ===== */
.preview-box{
    background:#050505;
    border:1px solid #222;
    border-radius:12px;
    padding:10px;
}

/* ===== TEXT GỐC ===== */
.preview-src{
    color:#facc15;
}

/* ===== TEXT DỊCH ===== */
.preview-dst{
    color:#22c55e;
    font-weight:600;
}

/* ===== METRIC ===== */
.metric-value{
    color:#facc15;
    font-weight:900;
}

/* ===== DOWNLOAD ===== */
.stDownloadButton > button{
    background:linear-gradient(90deg,#15803d,#22c55e);
    color:white;
    border:none;
    height:48px;
    font-weight:700;
}

</style>
"""


# =========================
# MODEL DỮ LIỆU
# =========================
@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""


# =========================
# HÀM SRT
# =========================
def read_srt_content(content: str) -> List[SubtitleItem]:
    content = content.strip()
    if not content:
        return []
    blocks = re.split(r"\n\s*\n", content)
    items: List[SubtitleItem] = []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        index = lines[0].strip()
        timecode = lines[1].strip()
        text = "\n".join(line.rstrip() for line in lines[2:]).strip()
        items.append(SubtitleItem(index=index, timecode=timecode, text=text))
    return items


def write_srt_content(items: List[SubtitleItem]) -> str:
    output = io.StringIO()
    for item in items:
        text = item.translated_text.strip() if item.translated_text.strip() else item.text.strip()
        output.write(f"{item.index}\n")
        output.write(f"{item.timecode}\n")
        output.write(f"{text}\n\n")
    return output.getvalue()


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def is_skip_line(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return True
    return not contains_chinese(sample)


def prepare_items(items: List[SubtitleItem]) -> Tuple[List[SubtitleItem], int]:
    skipped = 0
    for item in items:
        if is_skip_line(item.text):
            item.translated_text = item.text
            skipped += 1
    return items, skipped


def build_batches(items: List[SubtitleItem], batch_size: int) -> List[List[SubtitleItem]]:
    pending = [item for item in items if not item.translated_text.strip() and contains_chinese(item.text)]
    return [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]


# =========================
# GEMINI
# =========================
def create_client(api_key: str):
    return genai.Client(api_key=api_key.strip())


def build_prompt(batch: List[SubtitleItem], style_prompt: str) -> str:
    rows = []
    for i, item in enumerate(batch, start=1):
        clean_text = item.text.replace("\r", "").strip()
        rows.append(f"[{i}] {clean_text}")
    joined_rows = "\n".join(rows)
    extra = ""
    if style_prompt.strip():
        extra = f"\nYÊU CẦU PHONG CÁCH DỊCH RIÊNG:\n{style_prompt.strip()}\n"
    prompt = (
        f"{BASE_SYSTEM_PROMPT}\n"
        f"{extra}\n"
        "Hãy dịch danh sách phụ đề sau.\n"
        "Mỗi mục phải trả về đúng 1 dòng theo định dạng:\n"
        "[số] bản_dịch\n\n"
        "DANH SÁCH:\n"
        f"{joined_rows}"
    )
    return prompt.strip()


def parse_translated_response(batch: List[SubtitleItem], response_text: str) -> List[str]:
    mapping: Dict[int, str] = {}
    for line in response_text.splitlines():
        line = line.strip()
        match = re.match(r"^\[(\d+)\]\s*(.*)$", line)
        if match:
            idx = int(match.group(1))
            mapping[idx] = match.group(2).strip()
    results = []
    for i in range(1, len(batch) + 1):
        txt = mapping.get(i, "").strip()
        if not txt:
            txt = batch[i - 1].text
        results.append(txt)
    return results


def try_translate_batch_with_key(api_key: str, model_name: str, batch: List[SubtitleItem], style_prompt: str) -> List[str]:
    client = create_client(api_key)
    prompt = build_prompt(batch, style_prompt)
    last_error = None
    for _ in range(MAX_RETRIES_PER_KEY):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Model trả về rỗng")
            return parse_translated_response(batch, text)
        except Exception as e:
            last_error = e
            time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(str(last_error) if last_error else "Unknown error")


def translate_batch_with_failover(batch_id: int, batch: List[SubtitleItem], worker_slots: List[str], model_name: str, style_prompt: str):
    last_error = ""
    for api_key in worker_slots:
        try:
            translated = try_translate_batch_with_key(api_key, model_name, batch, style_prompt)
            return batch_id, True, translated, ""
        except Exception as e:
            last_error = str(e)
    return batch_id, False, [item.text for item in batch], last_error


# =========================
# SESSION STATE
# =========================
def init_state():
    defaults = {
        "translated_srt": "",
        "run_logs": [],
        "last_preview_src": "",
        "last_preview_dst": "",
        "finished": False,
        "had_error": False,
        "result_ready": False,
        "filename": "output_vi.srt",
        "stats": {
            "total": 0,
            "skip": 0,
            "need": 0,
            "ok_batches": 0,
            "failed_batches": 0,
            "done_lines": 0,
        },
        "elapsed_seconds": 0.0,
        "speed_text": "0 dòng/s",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# TOPBAR
# =========================
st.markdown(
    '<div class="topbar">'
    '  <div class="brand-wrap">'
    '    <div class="brand-icon">DT</div>'
    '    <div>'
    '      <div class="brand-title">Đình Thái</div>'
    '      <div class="brand-sub">SRT Translator Pro</div>'
    '    </div>'
    '  </div>'
    '  <div class="version-pill">V3.0 RED EDITION</div>'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1.55], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Nguồn Phụ Đề</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Kéo thả file .srt hoặc click để chọn", type=["srt"], label_visibility="collapsed")
    st.markdown('<div class="card-note">Hỗ trợ định dạng SubRip (.srt)</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚙️ Cấu Hình Dịch</div>', unsafe_allow_html=True)
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        model_name = st.selectbox("MODEL", options=["gemini-2.5-flash", "gemini-2.0-flash"], index=0)
    with cfg2:
        batch_size = st.number_input("BATCH SIZE", min_value=1, max_value=200, value=DEFAULT_BATCH_SIZE, step=1)
    output_name = st.text_input("TÊN FILE XUẤT", value="output_vi.srt")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✍️ Prompt Phong Cách Dịch</div>', unsafe_allow_html=True)
    style_prompt = st.text_area(
        "Prompt",
        value="Dịch tự nhiên, mượt như phim Trung. Xưng hô phù hợp ngữ cảnh, ưu tiên câu ngắn gọn, dễ đọc.",
        height=130,
        label_visibility="collapsed",
    )
    st.markdown('<div class="card-note">Phần thêm prompt vẫn giữ như cũ, bạn có thể nhập phong cách dịch riêng tại đây.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔑 Danh Sách API Key</div>', unsafe_allow_html=True)
    keys_text = st.text_area(
        "API Keys",
        placeholder="AIza...\nAIza...\nAIza...",
        height=150,
        label_visibility="collapsed",
    )
    batch_text = st.text_area(
        "Slots",
        value="1\n1\n1",
        height=92,
        help="Mỗi dòng là số slot tương ứng với từng API key",
    )
    st.markdown('<div class="card-note">Mỗi dòng 1 API key. Ô bên dưới là số slot tương ứng từng key. Tool sẽ tự retry khi key lỗi.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="ready-box">', unsafe_allow_html=True)
    st.markdown('<div class="ready-icon">▶</div>', unsafe_allow_html=True)
    st.markdown('<div class="ready-title">Sẵn Sàng</div>', unsafe_allow_html=True)
    st.markdown('<div class="ready-sub">Nhấn nút bên dưới để bắt đầu quá trình dịch tự động.</div>', unsafe_allow_html=True)
    st.markdown('<div class="mono-label">Tiến độ</div>', unsafe_allow_html=True)
    progress_placeholder = st.empty()
    start_col, clear_col = st.columns([5, 1])
    with start_col:
        run_btn = st.button("▶ Bắt Đầu Dịch", type="primary", use_container_width=True)
    with clear_col:
        clear_btn = st.button("🗑", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    stats = st.session_state["stats"]
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tổng dòng</div><div class="metric-value">{stats["total"]}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Cần dịch</div><div class="metric-value">{stats["need"]}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tốc độ</div><div class="metric-value">{st.session_state["speed_text"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🖥 Console Logs</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()
    if st.session_state["run_logs"]:
        log_placeholder.code("\n".join(st.session_state["run_logs"][-12:]))
    else:
        log_placeholder.info("Chưa có hoạt động nào...")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👀 Preview Dịch</div>', unsafe_allow_html=True)
    if st.session_state["last_preview_src"] or st.session_state["last_preview_dst"]:
        p1, p2 = st.columns(2)
        with p1:
            st.markdown('<div class="preview-title">Câu gốc</div>', unsafe_allow_html=True)
            st.code(st.session_state["last_preview_src"])
        with p2:
            st.markdown('<div class="preview-title">Bản dịch</div>', unsafe_allow_html=True)
            st.code(st.session_state["last_preview_dst"])
    else:
        st.info("Câu gốc và bản dịch mới nhất sẽ hiện ở đây.")
    st.markdown('</div>', unsafe_allow_html=True)


def collect_api_keys_and_slots(keys_raw: str, batches_raw: str) -> Tuple[List[str], List[str]]:
    raw_keys = keys_raw.splitlines()
    raw_batches = batches_raw.splitlines()
    api_keys = [line.strip() for line in raw_keys if line.strip()]
    batch_values = [line.strip() for line in raw_batches]
    worker_slots: List[str] = []
    for idx, key in enumerate(api_keys):
        batch_value = batch_values[idx] if idx < len(batch_values) else "1"
        try:
            count = int(batch_value)
            if count < 1:
                count = 1
        except ValueError:
            count = 1
        for _ in range(count):
            worker_slots.append(key)
    return api_keys, worker_slots


if clear_btn:
    st.session_state["translated_srt"] = ""
    st.session_state["run_logs"] = []
    st.session_state["last_preview_src"] = ""
    st.session_state["last_preview_dst"] = ""
    st.session_state["finished"] = False
    st.session_state["had_error"] = False
    st.session_state["result_ready"] = False
    st.session_state["elapsed_seconds"] = 0.0
    st.session_state["speed_text"] = "0 dòng/s"
    st.session_state["stats"] = {
        "total": 0,
        "skip": 0,
        "need": 0,
        "ok_batches": 0,
        "failed_batches": 0,
        "done_lines": 0,
    }
    st.rerun()


if run_btn:
    st.session_state["translated_srt"] = ""
    st.session_state["run_logs"] = []
    st.session_state["last_preview_src"] = ""
    st.session_state["last_preview_dst"] = ""
    st.session_state["finished"] = False
    st.session_state["had_error"] = False
    st.session_state["result_ready"] = False
    st.session_state["filename"] = output_name
    st.session_state["elapsed_seconds"] = 0.0
    st.session_state["speed_text"] = "0 dòng/s"

    if uploaded_file is None:
        st.error("Bạn chưa tải file SRT.")
    else:
        api_keys, worker_slots = collect_api_keys_and_slots(keys_text, batch_text)
        if not api_keys:
            st.error("Bạn chưa nhập API key nào.")
        elif not worker_slots:
            st.error("Không có worker slot hợp lệ.")
        else:
            content = uploaded_file.read().decode("utf-8-sig")
            items = read_srt_content(content)
            if not items:
                st.error("File SRT rỗng hoặc không đọc được.")
            else:
                items, skipped = prepare_items(items)
                batches = build_batches(items, int(batch_size))
                st.session_state["stats"] = {
                    "total": len(items),
                    "skip": skipped,
                    "need": sum(len(batch) for batch in batches),
                    "ok_batches": 0,
                    "failed_batches": 0,
                    "done_lines": 0,
                }
                progress_bar = progress_placeholder.progress(0.0)
                logs: List[str] = []
                total_batches = len(batches)
                start_time = time.time()

                if total_batches == 0:
                    st.session_state["translated_srt"] = write_srt_content(items)
                    st.session_state["result_ready"] = True
                    st.session_state["finished"] = True
                    log_placeholder.info("Không có dòng nào cần dịch. Tool đã bỏ qua các phần đã là tiếng Việt hoặc không có tiếng Trung.")
                else:
                    with ThreadPoolExecutor(max_workers=max(1, len(worker_slots))) as executor:
                        futures = []
                        for batch_id, batch in enumerate(batches):
                            futures.append(
                                executor.submit(
                                    translate_batch_with_failover,
                                    batch_id,
                                    batch,
                                    worker_slots,
                                    model_name,
                                    style_prompt,
                                )
                            )
                        completed = 0
                        done_lines = 0
                        for future in as_completed(futures):
                            batch_id, ok, translated_lines, error_text = future.result()
                            batch = batches[batch_id]
                            for item, translated in zip(batch, translated_lines):
                                item.translated_text = translated
                            st.session_state["last_preview_src"] = "\n\n".join(item.text for item in batch[:3])
                            st.session_state["last_preview_dst"] = "\n\n".join(item.translated_text for item in batch[:3])
                            completed += 1
                            done_lines += len(batch)
                            elapsed = max(time.time() - start_time, 0.001)
                            speed = done_lines / elapsed
                            st.session_state["elapsed_seconds"] = elapsed
                            st.session_state["speed_text"] = f"{speed:.1f} dòng/s"
                            st.session_state["stats"]["done_lines"] = done_lines
                            progress_bar.progress(completed / total_batches)
                            if ok:
                                st.session_state["stats"]["ok_batches"] += 1
                                logs.append(f"✓ Batch {batch_id + 1}/{total_batches} dịch xong | {len(batch)} dòng | {speed:.1f} dòng/s")
                            else:
                                st.session_state["stats"]["failed_batches"] += 1
                                st.session_state["had_error"] = True
                                logs.append(f"✗ Batch {batch_id + 1}/{total_batches} lỗi: {error_text}")
                            st.session_state["run_logs"] = logs
                            log_placeholder.code("\n".join(logs[-12:]))
                    st.session_state["translated_srt"] = write_srt_content(items)
                    st.session_state["result_ready"] = True
                    st.session_state["finished"] = True


if st.session_state["finished"]:
    if st.session_state["had_error"]:
        st.markdown('<div class="warn-box">Đã dịch xong phần thành công. Một số batch vẫn lỗi. Bạn có thể tải file hiện tại và chạy lại với thêm API key ổn định hơn để xử lý phần còn lỗi.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ok-box">Dịch xong rồi. Bạn có thể tải file SRT ngay bên dưới.</div>', unsafe_allow_html=True)

if st.session_state["result_ready"] and st.session_state["translated_srt"]:
    st.download_button(
        label="Tải file SRT đã dịch",
        data=st.session_state["translated_srt"].encode("utf-8"),
        file_name=st.session_state["filename"],
        mime="application/x-subrip",
        use_container_width=True,
    )
