import io
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from google import genai
from google.genai import types


# ============================================================
# CẤU HÌNH
# ============================================================
st.set_page_config(
    page_title="Đình Thái - Dịch SRT Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 40
MAX_RETRIES_PER_KEY = 2
REQUEST_DELAY = 0.35

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
    padding-top: 1.4rem;
    padding-bottom: 1.5rem;
    max-width: 1320px;
}
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #94a3b8;
    margin-bottom: 1rem;
}
.card {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 18px;
    padding: 18px 18px 12px 18px;
    margin-bottom: 14px;
}
.metric-card {
    background: #0f172a;
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 14px 16px;
    min-height: 94px;
}
.metric-label {
    color: #94a3b8;
    font-size: 0.9rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    margin-top: 4px;
}
.small-note {
    color: #94a3b8;
    font-size: 0.9rem;
}
.good-box {
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 14px;
    padding: 12px 14px;
}
.warn-box {
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 14px;
    padding: 12px 14px;
}
</style>
"""


# ============================================================
# MODEL DỮ LIỆU
# ============================================================
@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""
    done: bool = False


# ============================================================
# XỬ LÝ SRT
# ============================================================
def read_srt_content(content: str) -> List[SubtitleItem]:
    content = content.strip()
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
        final_text = item.translated_text.strip() if item.translated_text.strip() else item.text.strip()
        output.write(f"{item.index}\n")
        output.write(f"{item.timecode}\n")
        output.write(f"{final_text}\n\n")
    return output.getvalue()


def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def is_already_translated(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return True
    return not contains_chinese(sample)


def split_batches(items: List[SubtitleItem], batch_size: int) -> List[List[SubtitleItem]]:
    pending = [item for item in items if not item.done and not item.translated_text.strip() and contains_chinese(item.text)]
    return [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]


# ============================================================
# GEMINI
# ============================================================
def create_client(api_key: str):
    return genai.Client(api_key=api_key.strip())


def build_prompt(batch: List[SubtitleItem], style_prompt: str) -> str:
    rows = []
    for i, item in enumerate(batch, start=1):
        rows.append(f"[{i}] {item.text.replace(chr(13), '').strip()}")

    extra = ""
    if style_prompt.strip():
        extra = f"\nYÊU CẦU PHONG CÁCH DỊCH RIÊNG:\n{style_prompt.strip()}\n"

    return f"""
{BASE_SYSTEM_PROMPT}
{extra}
Hãy dịch danh sách phụ đề sau.
Mỗi mục phải trả về đúng 1 dòng theo định dạng:
[số] bản_dịch

DANH SÁCH:
{"\n".join(rows)}
""".strip()


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


def try_translate_batch_with_key(
    api_key: str,
    model_name: str,
    batch: List[SubtitleItem],
    style_prompt: str,
) -> List[str]:
    client = create_client(api_key)
    prompt = build_prompt(batch, style_prompt)
    last_error: Optional[Exception] = None

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
            time.sleep(REQUEST_DELAY)

    raise RuntimeError(str(last_error) if last_error else "Unknown error")


def translate_batch_with_failover(
    batch_id: int,
    batch: List[SubtitleItem],
    worker_slots: List[str],
    model_name: str,
    style_prompt: str,
) -> Tuple[int, bool, List[str], str]:
    last_error = ""

    for api_key in worker_slots:
        try:
            translated = try_translate_batch_with_key(api_key, model_name, batch, style_prompt)
            return batch_id, True, translated, ""
        except Exception as e:
            last_error = str(e)
            continue

    return batch_id, False, [item.text for item in batch], last_error


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "items": [],
        "translated_srt": "",
        "run_logs": [],
        "last_preview": "",
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
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# UI
# ============================================================
init_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Đình Thái - Dịch SRT Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Layout đẹp • Nhiều API key • Retry key lỗi • Preview câu dịch • Progress realtime • Download SRT</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.7, 1.05], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Tải file SRT", type=["srt"])
    style_prompt = st.text_area(
        "Prompt phong cách dịch",
        value="Dịch tự nhiên, mượt như phim Trung. Xưng hô phù hợp ngữ cảnh, ưu tiên câu ngắn gọn, dễ đọc.",
        height=120,
    )
    keys_text = st.text_area(
        "Danh sách API key (mỗi dòng 1 key)",
        placeholder="AIza...\nAIza...\nAIza...",
        height=150,
    )
    st.markdown('<div class="small-note">Dán nhiều API key xuống dòng để tool tự xoay vòng khi gặp key lỗi.</div>', unsafe_allow_html=True)
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

    run_col, clear_col = st.columns([2, 1])
    with run_col:
        run_btn = st.button("Bắt đầu dịch", type="primary", use_container_width=True)
    with clear_col:
        clear_btn = st.button("Xóa kết quả", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    stats = st.session_state["stats"]
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Tổng dòng</div><div class="metric-value">{stats["total"]}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Cần dịch</div><div class="metric-value">{stats["need"]}</div></div>', unsafe_allow_html=True)

    s3, s4 = st.columns(2)
    with s3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Bỏ qua</div><div class="metric-value">{stats["skip"]}</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Batch lỗi</div><div class="metric-value">{stats["failed_batches"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preview dịch")
    preview_placeholder = st.empty()
    if st.session_state["last_preview"]:
        preview_placeholder.code(st.session_state["last_preview"])
    else:
        preview_placeholder.info("Câu gốc và câu đã dịch sẽ hiện ở đây.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Tiến trình")
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    if st.session_state["run_logs"]:
        log_placeholder.code("\n".join(st.session_state["run_logs"][-12:]))
    else:
        log_placeholder.info("Chưa có log.")
    st.markdown('</div>', unsafe_allow_html=True)

if clear_btn:
    st.session_state["items"] = []
    st.session_state["translated_srt"] = ""
    st.session_state["run_logs"] = []
    st.session_state["last_preview"] = ""
    st.session_state["finished"] = False
    st.session_state["had_error"] = False
    st.session_state["result_ready"] = False
    st.session_state["stats"] = {
        "total": 0,
        "skip": 0,
        "need": 0,
        "ok_batches": 0,
        "failed_batches": 0,
    }
    st.rerun()


# ============================================================
# HÀM HỖ TRỢ RUN
# ============================================================
def collect_api_keys_and_slots(keys_text: str, batch_text: str):
    raw_keys = keys_text.splitlines()
    raw_batches = batch_text.splitlines()

    api_keys = [line.strip() for line in raw_keys if line.strip()]
    batch_values = [line.strip() for line in raw_batches]

    worker_slots = []
    for idx, key in enumerate(api_keys):
        batch_value = batch_values[idx].strip() if idx < len(batch_values) else "1"
        try:
            count = int(batch_value)
            if count < 1:
                count = 1
        except ValueError:
            count = 1
        for _ in range(count):
            worker_slots.append(key)

    return api_keys, worker_slots


def apply_existing_translations_if_any(items: List[SubtitleItem]):
    skipped = 0
    for item in items:
        if is_already_translated(item.text):
            item.translated_text = item.text
            item.done = True
            skipped += 1
    return skipped


# ============================================================
# XỬ LÝ CHẠY
# ============================================================
if run_btn:
    st.session_state["finished"] = False
    st.session_state["had_error"] = False
    st.session_state["result_ready"] = False
    st.session_state["run_logs"] = []
    st.session_state["last_preview"] = ""
    st.session_state["filename"] = output_name

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
                skipped = apply_existing_translations_if_any(items)
                batches = split_batches(items, int(batch_size))

                st.session_state["stats"] = {
                    "total": len(items),
                    "skip": skipped,
                    "need": sum(len(b) for b in batches),
                    "ok_batches": 0,
                    "failed_batches": 0,
                }

                progress_bar = progress_placeholder.progress(0.0)
                logs = []
                total_batches = len(batches)

                if total_batches == 0:
                    result_srt = write_srt_content(items)
                    st.session_state["translated_srt"] = result_srt
                    st.session_state["items"] = items
                    st.session_state["result_ready"] = True
                    st.session_state["finished"] = True
                    preview_placeholder.info("Không có dòng nào cần dịch. Tool đã bỏ qua các phần đã là tiếng Việt hoặc không có tiếng Trung.")
                else:
                    futures = []
                    with ThreadPoolExecutor(max_workers=max(1, len(worker_slots))) as executor:
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
                        for future in as_completed(futures):
                            batch_id, ok, translated_lines, error_text = future.result()
                            batch = batches[batch_id]

                            for item, translated in zip(batch, translated_lines):
                                item.translated_text = translated
                                item.done = True

                            preview_text = "\n\n".join(
                                f"Gốc: {item.text}\nDịch: {item.translated_text}"
                                for item in batch[:3]
                            )
                            st.session_state["last_preview"] = preview_text
                            preview_placeholder.code(preview_text)

                            completed += 1
                            progress_bar.progress(completed / total_batches)

                            if ok:
                                st.session_state["stats"]["ok_batches"] += 1
                                logs.append(f"✓ Batch {batch_id + 1}/{total_batches} dịch xong")
                            else:
                                st.session_state["stats"]["failed_batches"] += 1
                                st.session_state["had_error"] = True
                                logs.append(f"✗ Batch {batch_id + 1}/{total_batches} lỗi: {error_text}")

                            st.session_state["run_logs"] = logs
                            log_placeholder.code("\n".join(logs[-12:]))

                    result_srt = write_srt_content(items)
                    st.session_state["translated_srt"] = result_srt
                    st.session_state["items"] = items
                    st.session_state["result_ready"] = True
                    st.session_state["finished"] = True


# ============================================================
# KẾT QUẢ
# ============================================================
if st.session_state["finished"]:
    if st.session_state["had_error"]:
        st.markdown('<div class="warn-box">Đã dịch xong phần thành công. Một số batch bị lỗi. Khi bạn bấm dịch lại với cùng file, tool sẽ chỉ xử lý lại phần còn lỗi và giữ nguyên phần đã dịch thành công.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="good-box">Dịch xong rồi. Bạn có thể tải file SRT ngay bên dưới.</div>', unsafe_allow_html=True)

if st.session_state["result_ready"] and st.session_state["translated_srt"]:
    st.download_button(
        label="Tải file SRT đã dịch",
        data=st.session_state["translated_srt"].encode("utf-8"),
        file_name=st.session_state["filename"],
        mime="application/x-subrip",
        use_container_width=True,
    )
