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
    page_title="Đình Thái - SRT Translator Pro V5",
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
.block-container {
    max-width: 1400px;
    padding-top: 0.8rem;
    padding-bottom: 1.4rem;
}
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #f6f7fb 0%, #eef2f7 100%);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.topbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    background: rgba(255,255,255,0.76);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(15,23,42,0.06);
    box-shadow: 0 10px 30px rgba(15,23,42,0.06);
    border-radius: 22px;
    padding: 16px 18px;
    margin-bottom: 18px;
}
.brand-wrap {
    display:flex;
    align-items:center;
    gap:14px;
}
.brand-icon {
    width:44px;
    height:44px;
    border-radius:14px;
    background: linear-gradient(135deg, #2563eb 0%, #60a5fa 100%);
    color:#fff;
    display:flex;
    align-items:center;
    justify-content:center;
    font-weight:800;
    box-shadow: 0 10px 25px rgba(37,99,235,0.25);
}
.brand-title {
    font-size:1.95rem;
    line-height:1;
    font-weight:900;
    color:#0f172a;
}
.brand-sub {
    color:#64748b;
    letter-spacing:0.16em;
    text-transform:uppercase;
    font-size:0.82rem;
    margin-top:4px;
}
.version-pill {
    border: 1px solid rgba(15,23,42,0.08);
    background:#ffffff;
    color:#475569;
    border-radius:999px;
    padding:8px 14px;
    font-size:0.82rem;
    font-weight:700;
}
.card {
    background: rgba(255,255,255,0.88);
    border:1px solid rgba(15,23,42,0.06);
    border-radius:24px;
    padding:18px 18px 14px 18px;
    margin-bottom:18px;
    box-shadow: 0 10px 30px rgba(15,23,42,0.06);
}
.card-title {
    color:#0f172a;
    font-size:1.06rem;
    font-weight:800;
    margin-bottom:12px;
}
.card-note {
    color:#64748b;
    font-size:0.92rem;
}
.ready-box {
    min-height: 360px;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    text-align:center;
}
.ready-icon {
    width:84px;
    height:84px;
    border-radius:50%;
    background:#eef4ff;
    border:1px solid rgba(37,99,235,0.12);
    color:#2563eb;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:2rem;
    margin-bottom:16px;
}
.ready-title {
    color:#0f172a;
    font-size:2rem;
    font-weight:900;
    margin-bottom:8px;
}
.ready-sub {
    color:#64748b;
    margin-bottom:20px;
}
.mono-label {
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:0.16em;
    font-size:0.72rem;
    margin-bottom:8px;
}
.metric-grid {
    display:grid;
    grid-template-columns: repeat(3, 1fr);
    gap:12px;
}
.metric-card {
    background:#ffffff;
    border:1px solid rgba(15,23,42,0.06);
    border-radius:18px;
    padding:14px;
}
.metric-label {
    color:#64748b;
    font-size:0.84rem;
}
.metric-value {
    color:#0f172a;
    font-size:1.42rem;
    font-weight:900;
    margin-top:4px;
}
.metric-sub {
    color:#94a3b8;
    font-size:0.8rem;
    margin-top:3px;
}
.status-ok {
    background:#ecfdf5;
    border:1px solid #bbf7d0;
    color:#166534;
    border-radius:14px;
    padding:12px 14px;
    margin-top:8px;
}
.status-warn {
    background:#fff7ed;
    border:1px solid #fed7aa;
    color:#9a3412;
    border-radius:14px;
    padding:12px 14px;
    margin-top:8px;
}
.console-box {
    background:#f8fafc;
    border:1px solid rgba(15,23,42,0.06);
    border-radius:16px;
    padding:10px;
}
.preview-title {
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:0.14em;
    font-size:0.72rem;
    margin-bottom:8px;
}
div[data-testid="stFileUploaderDropzone"] {
    background:#f8fafc;
    border:2px dashed #cbd5e1;
    border-radius:20px;
}
.stTextInput > div > div > input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background:#ffffff !important;
    color:#0f172a !important;
    border:1px solid #dbe2ea !important;
    border-radius:14px !important;
}
.stButton > button {
    border-radius:16px !important;
    min-height:52px !important;
    font-weight:800 !important;
    border:1px solid rgba(15,23,42,0.08) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
    color:white !important;
    box-shadow: 0 12px 24px rgba(37,99,235,0.2);
}
.stDownloadButton > button {
    border-radius:16px !important;
    min-height:52px !important;
    font-weight:800 !important;
    background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%) !important;
    color:white !important;
    border:none !important;
}
[data-testid="stProgressBar"] > div {
    background:#e2e8f0 !important;
}
[data-testid="stProgressBar"] div div div {
    background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%) !important;
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
        "speed_text": "0 dòng/s",
        "progress_percent": 0,
        "stop_requested": False,
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
    '  <div class="version-pill">V5.0 LIGHT EDITION</div>'
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
    st.markdown('<div class="card-note">Bạn có thể nhập phong cách dịch riêng tại đây.</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="card-note">Mỗi dòng 1 API key. Ô bên dưới là số slot tương ứng từng key.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="ready-box">', unsafe_allow_html=True)
    st.markdown('<div class="ready-icon">▶</div>', unsafe_allow_html=True)
    st.markdown('<div class="ready-title">Sẵn Sàng</div>', unsafe_allow_html=True)
    st.markdown('<div class="ready-sub">Nhấn nút bên dưới để bắt đầu quá trình dịch tự động.</div>', unsafe_allow_html=True)
    st.markdown('<div class="mono-label">Tiến độ</div>', unsafe_allow_html=True)
    progress_placeholder = st.empty()
    percent_placeholder = st.empty()
    b1, b2 = st.columns([5, 1])
    with b1:
        run_btn = st.button("▶ Bắt Đầu Dịch", type="primary", use_container_width=True)
    with b2:
        stop_btn = st.button("⏹", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    stats = st.session_state["stats"]
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tổng dòng</div><div class="metric-value">{stats["total"]}</div><div class="metric-sub">Toàn bộ subtitle</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Cần dịch</div><div class="metric-value">{stats["need"]}</div><div class="metric-sub">Chỉ các dòng còn tiếng Trung</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tốc độ</div><div class="metric-value">{st.session_state["speed_text"]}</div><div class="metric-sub">Ước tính hiện tại</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🖥 Console Logs</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()
    if st.session_state["run_logs"]:
        log_placeholder.code("\n".join(st.session_state["run_logs"][-12:]))
    else:
        log_placeholder.info("Đang chờ bạn tải file SRT, dán API key và bấm Bắt đầu dịch.")
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


if stop_btn:
    st.session_state["stop_requested"] = True
    st.warning("Đã nhận lệnh dừng. Tool sẽ dừng sau batch hiện tại.")


if run_btn:
    st.session_state["translated_srt"] = ""
    st.session_state["run_logs"] = []
    st.session_state["last_preview_src"] = ""
    st.session_state["last_preview_dst"] = ""
    st.session_state["finished"] = False
    st.session_state["had_error"] = False
    st.session_state["result_ready"] = False
    st.session_state["filename"] = output_name
    st.session_state["speed_text"] = "0 dòng/s"
    st.session_state["progress_percent"] = 0
    st.session_state["stop_requested"] = False

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
                percent_placeholder.markdown(f"**0%**")
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
                            if st.session_state.get("stop_requested", False):
                                logs.append("⏹ Đã dừng theo yêu cầu người dùng.")
                                st.session_state["run_logs"] = logs
                                log_placeholder.code("\n".join(logs[-12:]))
                                break

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
                            percent = int((completed / total_batches) * 100)
                            st.session_state["speed_text"] = f"{speed:.1f} dòng/s"
                            st.session_state["progress_percent"] = percent
                            st.session_state["stats"]["done_lines"] = done_lines
                            progress_bar.progress(completed / total_batches)
                            percent_placeholder.markdown(f"**{percent}%**")

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
    if st.session_state["stop_requested"]:
        st.markdown('<div class="status-warn">Tool đã dừng theo yêu cầu. Những phần đã dịch xong vẫn được giữ lại và bạn có thể tải file hiện tại.</div>', unsafe_allow_html=True)
    elif st.session_state["had_error"]:
        st.markdown('<div class="status-warn">Đã dịch xong phần thành công. Một số batch vẫn lỗi. Bạn có thể tải file hiện tại và chạy lại với thêm API key ổn định hơn.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-ok">Dịch xong rồi. Bạn có thể tải file SRT ngay bên dưới.</div>', unsafe_allow_html=True)

if st.session_state["result_ready"] and st.session_state["translated_srt"]:
    st.download_button(
        label="Tải file SRT đã dịch",
        data=st.session_state["translated_srt"].encode("utf-8"),
        file_name=st.session_state["filename"],
        mime="application/x-subrip",
        use_container_width=True,
    )
