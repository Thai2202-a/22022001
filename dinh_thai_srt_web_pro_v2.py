import io
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google import genai
from google.genai import types


# =========================
# CẤU HÌNH CHUNG
# =========================
st.set_page_config(
    page_title="Đình Thái Studio - Dịch SRT Pro V2",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 28
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
    max-width: 1380px;
    padding-top: 1.1rem;
    padding-bottom: 1.8rem;
}
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.app-shell {
    background: radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(168,85,247,0.16), transparent 22%),
                linear-gradient(180deg, #020617 0%, #0b1120 100%);
    border: 1px solid rgba(148,163,184,0.16);
    border-radius: 26px;
    padding: 20px 22px 10px 22px;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 2.4rem;
    line-height: 1.1;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: #94a3b8;
    margin-bottom: 0.9rem;
}
.card {
    background: linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(17,24,39,0.98) 100%);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 22px;
    padding: 18px 18px 12px 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.18);
}
.metric-card {
    background: linear-gradient(180deg, rgba(15,23,42,1) 0%, rgba(17,24,39,1) 100%);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 18px;
    padding: 14px 16px;
    min-height: 96px;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.9rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 900;
    margin-top: 4px;
}
.metric-sub {
    color: #64748b;
    font-size: 0.84rem;
    margin-top: 4px;
}
.note {
    color: #94a3b8;
    font-size: 0.92rem;
}
.ok-box {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.26);
    border-radius: 14px;
    padding: 12px 14px;
}
.warn-box {
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.26);
    border-radius: 14px;
    padding: 12px 14px;
}
.badge-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.badge {
    background: rgba(59,130,246,0.14);
    color: #bfdbfe;
    border: 1px solid rgba(59,130,246,0.25);
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 0.84rem;
}
.dual-preview {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.preview-box {
    background: rgba(2,6,23,0.66);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 14px;
    padding: 12px;
}
.preview-title {
    color: #94a3b8;
    font-size: 0.82rem;
    margin-bottom: 8px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
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
# GIAO DIỆN
# =========================
st.markdown('<div class="app-shell">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🎬 Đình Thái Studio - Dịch SRT Pro V2</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Bản PRO đẹp hơn: progress realtime, preview song song Trung → Việt, tốc độ dịch/giây, auto giữ phần đã xong, nhiều API key và retry key lỗi.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="badge-row">'
    '<span class="badge">Nhiều API key</span>'
    '<span class="badge">Retry key lỗi</span>'
    '<span class="badge">Progress realtime</span>'
    '<span class="badge">Preview song song</span>'
    '<span class="badge">Tốc độ dịch / giây</span>'
    '<span class="badge">Tải SRT sau khi xong</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

left, right = st.columns([1.7, 1.08], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Tải file SRT", type=["srt"])
    style_prompt = st.text_area(
        "Prompt phong cách dịch",
        value="Dịch tự nhiên, mượt như phim Trung. Xưng hô phù hợp ngữ cảnh, ưu tiên câu ngắn gọn, dễ đọc.",
        height=130,
    )
    keys_text = st.text_area(
        "Danh sách API key (mỗi dòng 1 key)",
        placeholder="AIza...\nAIza...\nAIza...",
        height=160,
    )
    st.markdown('<div class="note">Dán nhiều API key xuống dòng để tool tự đổi key nếu một key lỗi hoặc chậm.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        model_name = st.text_input("Model", value=DEFAULT_MODEL)
    with c2:
        batch_size = st.number_input("Batch size", min_value=1, max_value=200, value=DEFAULT_BATCH_SIZE, step=1)
    with c3:
        output_name = st.text_input("Tên file xuất", value="output_vi.srt")

    batch_text = st.text_area(
        "Batch / key (mỗi dòng 1 số tương ứng với từng key)",
        value="2\n1\n3",
        height=110,
    )

    btn1, btn2 = st.columns([2, 1])
    with btn1:
        run_btn = st.button("Bắt đầu dịch", type="primary", use_container_width=True)
    with btn2:
        clear_btn = st.button("Xóa kết quả", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    stats = st.session_state["stats"]
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tổng dòng</div><div class="metric-value">{stats["total"]}</div><div class="metric-sub">Toàn bộ subtitle</div></div>', unsafe_allow_html=True)
    with r1c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Cần dịch</div><div class="metric-value">{stats["need"]}</div><div class="metric-sub">Chỉ các dòng còn tiếng Trung</div></div>', unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Bỏ qua</div><div class="metric-value">{stats["skip"]}</div><div class="metric-sub">Dòng đã Việt hóa / không cần dịch</div></div>', unsafe_allow_html=True)
    with r2c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Batch lỗi</div><div class="metric-value">{stats["failed_batches"]}</div><div class="metric-sub">Sẽ giữ log để chạy lại</div></div>', unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Đã dịch xong</div><div class="metric-value">{stats["done_lines"]}</div><div class="metric-sub">Dòng hoàn tất trong lượt này</div></div>', unsafe_allow_html=True)
    with r3c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tốc độ</div><div class="metric-value">{st.session_state["speed_text"]}</div><div class="metric-sub">Ước tính theo thời gian chạy</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preview song song Trung → Việt")
    if st.session_state["last_preview_src"] or st.session_state["last_preview_dst"]:
        st.markdown('<div class="dual-preview">', unsafe_allow_html=True)
        st.markdown('<div class="preview-box"><div class="preview-title">Câu gốc</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="preview-box"><div class="preview-title">Bản dịch</div></div>', unsafe_allow_html=True)
        src_col, dst_col = st.columns(2)
        with src_col:
            st.code(st.session_state["last_preview_src"])
        with dst_col:
            st.code(st.session_state["last_preview_dst"])
    else:
        st.info("Câu gốc và câu đã dịch sẽ hiện ở đây trong quá trình chạy.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Tiến trình")
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    if st.session_state["run_logs"]:
        log_placeholder.code("\n".join(st.session_state["run_logs"][-12:]))
    else:
        log_placeholder.info("Đang chờ bạn tải file SRT, dán API key và bấm 'Bắt đầu dịch'.")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# HÀM PHỤ UI
# =========================
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


# =========================
# XỬ LÝ CHẠY
# =========================
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


# =========================
# KẾT QUẢ
# =========================
if st.session_state["finished"]:
    if st.session_state["had_error"]:
        st.markdown(
            '<div class="warn-box">Đã dịch xong phần thành công. Một số batch vẫn lỗi. Bạn có thể tải file hiện tại và chạy lại với thêm API key ổn định hơn để xử lý phần còn lỗi.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ok-box">Dịch xong rồi. Bạn có thể tải file SRT ngay bên dưới.</div>',
            unsafe_allow_html=True,
        )

if st.session_state["result_ready"] and st.session_state["translated_srt"]:
    st.download_button(
        label="Tải file SRT đã dịch",
        data=st.session_state["translated_srt"].encode("utf-8"),
        file_name=st.session_state["filename"],
        mime="application/x-subrip",
        use_container_width=True,
    )
