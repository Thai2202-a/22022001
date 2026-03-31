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
    page_title="Đình Thái - SRT Translator Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_BATCH_SIZE = 80
MAX_RETRIES_PER_KEY = 2
RETRY_SLEEP_SECONDS = 0.35
MAX_PARALLEL_BATCHES = 4

BASE_SYSTEM_PROMPT = """
Bạn là chuyên gia dịch phụ đề phim chuyên nghiệp.

YÊU CẦU CHUNG:
- Dịch tự nhiên, mượt, đúng ngữ cảnh hội thoại.
- Giữ văn phong giống phụ đề phim, dễ đọc, gọn.
- Không giải thích, không ghi chú.
- Không bỏ dòng nào.
- Không đánh số lại.
- Mỗi mục phụ đề đầu vào phải trả về đúng 1 dòng đầu ra tương ứng.
- Nếu gặp tên riêng thì xử lý hợp lý theo ngữ cảnh.
- Không thêm ký tự thừa.
""".strip()

SOURCE_LANGUAGE_OPTIONS = [
    "Tự động",
    "Tiếng Trung",
    "Tiếng Anh",
    "Tiếng Nhật",
    "Tiếng Hàn",
    "Tiếng Thái",
    "Tiếng Pháp",
    "Tiếng Đức",
    "Tiếng Nga",
    "Tiếng Tây Ban Nha",
    "Tiếng Bồ Đào Nha",
    "Tiếng Ả Rập",
]

# THÊM: Danh sách ngôn ngữ đích
TARGET_LANGUAGE_OPTIONS = [
    "Tiếng Việt",
    "Tiếng Anh",
    "Tiếng Trung (Phồn thể)",
    "Tiếng Trung (Giản thể)",
    "Tiếng Nhật",
    "Tiếng Hàn",
    "Tiếng Pháp",
    "Tiếng Thái",
    "Giống ngôn ngữ nguồn (Sửa lỗi)"
]

LANGUAGE_LABELS = {
    "zh": "Tiếng Trung",
    "ja": "Tiếng Nhật",
    "ko": "Tiếng Hàn",
    "th": "Tiếng Thái",
    "ru": "Tiếng Nga",
    "ar": "Tiếng Ả Rập",
    "vi": "Tiếng Việt",
    "es": "Tiếng Tây Ban Nha",
    "fr": "Tiếng Pháp",
    "de": "Tiếng Đức",
    "pt": "Tiếng Bồ Đào Nha",
    "en_or_latin": "Tiếng Anh / ngôn ngữ Latin",
    "unknown": "Không xác định",
}

CUSTOM_CSS = """
<style>
.block-container {max-width: 1450px; padding-top: .8rem; padding-bottom: 1.2rem;}
html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
[data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #f7f8fb 0%, #eef2f7 100%);}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}

.topbar {
    display:flex; justify-content:space-between; align-items:center;
    background:rgba(255,255,255,.86); backdrop-filter: blur(12px);
    border:1px solid rgba(15,23,42,.06); box-shadow:0 10px 30px rgba(15,23,42,.06);
    border-radius:24px; padding:16px 18px; margin-bottom:18px;
}
.brand-wrap {display:flex; align-items:center; gap:14px;}
.brand-icon {
    width:46px; height:46px; border-radius:14px;
    background:linear-gradient(135deg,#2563eb 0%,#60a5fa 100%);
    color:#fff; display:flex; align-items:center; justify-content:center;
    font-weight:900; box-shadow:0 10px 25px rgba(37,99,235,.25);
}
.brand-title {font-size:1.95rem; line-height:1; font-weight:900; color:#0f172a;}
.brand-sub {color:#64748b; letter-spacing:.16em; text-transform:uppercase; font-size:.82rem; margin-top:4px;}
.version-pill {
    border:1px solid rgba(15,23,42,.08); background:#fff; color:#475569;
    border-radius:999px; padding:8px 14px; font-size:.82rem; font-weight:700;
}

.card {
    background:rgba(255,255,255,.92); border:1px solid rgba(15,23,42,.06);
    border-radius:24px; padding:18px 18px 14px 18px; margin-bottom:18px;
    box-shadow:0 10px 30px rgba(15,23,42,.06);
}
.card-title {color:#0f172a; font-size:1.06rem; font-weight:800; margin-bottom:12px;}
.card-note {color:#64748b; font-size:.92rem;}

.metric-card {
    background:#fff; border:1px solid rgba(15,23,42,.06);
    border-radius:18px; padding:14px;
}
.metric-label {color:#64748b; font-size:.84rem;}
.metric-value {color:#0f172a; font-size:1.42rem; font-weight:900; margin-top:4px;}
.metric-sub {color:#94a3b8; font-size:.8rem; margin-top:3px;}

.stTextInput > div > div > input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background:#fff !important; color:#0f172a !important;
    border:1px solid #dbe2ea !important; border-radius:14px !important;
}

.stButton > button {
    border-radius:16px !important; min-height:52px !important;
    font-weight:800 !important; border:1px solid rgba(15,23,42,.08) !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(90deg,#2563eb 0%,#3b82f6 100%) !important;
    color:white !important; box-shadow:0 12px 24px rgba(37,99,235,.2);
}
.stDownloadButton > button {
    border-radius:16px !important; min-height:52px !important; font-weight:800 !important;
    background:linear-gradient(90deg,#16a34a 0%,#22c55e 100%) !important;
    color:white !important; border:none !important;
}

[data-testid="stProgressBar"] > div {background:#e2e8f0 !important;}
[data-testid="stProgressBar"] div div div {
    background:linear-gradient(90deg,#16a34a 0%,#22c55e 100%) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background:#f8fafc; border:2px dashed #cbd5e1; border-radius:20px;
}

.status-ok {
    background:#ecfdf5; border:1px solid #bbf7d0; color:#166534;
    border-radius:14px; padding:12px 14px; margin-top:8px;
}
.status-warn {
    background:#fff7ed; border:1px solid #fed7aa; color:#9a3412;
    border-radius:14px; padding:12px 14px; margin-top:8px;
}
.small {color:#64748b; font-size:.88rem;}

.key-box {
    border-radius:14px; padding:12px 14px; margin-bottom:10px;
    border:1px solid rgba(15,23,42,.08); background:#fff;
}
.key-row {
    display:flex; justify-content:space-between; align-items:center;
    gap:12px; flex-wrap:wrap;
}
.key-name {font-weight:700; color:#0f172a; font-size:.95rem; word-break:break-all;}
.key-status {padding:6px 10px; border-radius:999px; font-size:.8rem; font-weight:800;}
.key-green {background:#ecfdf5; color:#166534; border:1px solid #bbf7d0;}
.key-red {background:#fef2f2; color:#991b1b; border:1px solid #fecaca;}
.key-yellow {background:#fffbeb; color:#92400e; border:1px solid #fde68a;}
.key-detail {margin-top:8px; font-size:.84rem; color:#64748b; white-space:pre-wrap; word-break:break-word;}

.done-popup {
    position: fixed;
    right: 24px;
    bottom: 24px;
    width: 360px;
    max-width: calc(100vw - 32px);
    background: rgba(255,255,255,.98);
    border: 1px solid rgba(15,23,42,.08);
    box-shadow: 0 18px 40px rgba(15,23,42,.16);
    border-radius: 22px;
    z-index: 99999;
    overflow: hidden;
}
.done-popup-body {
    padding: 16px;
}
.done-check {
    width: 56px;
    height: 56px;
    border-radius: 999px;
    background: linear-gradient(135deg,#16a34a 0%,#22c55e 100%);
    color: white;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size: 30px;
    font-weight: 900;
    box-shadow: 0 12px 24px rgba(34,197,94,.28);
    margin-bottom: 12px;
}
.done-title {
    font-size: 1.16rem;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 6px;
}
.done-text {
    color: #64748b;
    font-size: .95rem;
    line-height: 1.5;
}

.translate-inline {
    background:#eff6ff;
    border:1px solid #bfdbfe;
    color:#1d4ed8;
    border-radius:18px;
    padding:14px 16px;
    display:flex;
    align-items:center;
    gap:12px;
    margin-bottom:16px;
}
.translate-inline-spinner {
    width:20px;
    height:20px;
    border-radius:999px;
    border:3px solid #bfdbfe;
    border-top-color:#2563eb;
    animation: spinDtInline .8s linear infinite;
    flex:0 0 auto;
}
.translate-inline-title {
    font-weight:900;
    color:#1e3a8a;
    font-size:.96rem;
}
.translate-inline-text {
    color:#1d4ed8;
    font-size:.9rem;
    margin-top:2px;
}

.batch-live-box {
    background:#fff;
    border:1px solid rgba(15,23,42,.06);
    border-radius:18px;
    padding:14px;
    min-height:220px;
}
.batch-line-pending {
    padding:10px 12px;
    border-radius:12px;
    background:#eff6ff;
    color:#1d4ed8;
    margin-bottom:8px;
    border:1px solid #bfdbfe;
    font-weight:700;
}
.batch-line-done {
    padding:10px 12px;
    border-radius:12px;
    background:#ecfdf5;
    color:#166534;
    margin-bottom:8px;
    border:1px solid #bbf7d0;
    font-weight:700;
}
.batch-line-error {
    padding:10px 12px;
    border-radius:12px;
    background:#fef2f2;
    color:#991b1b;
    margin-bottom:8px;
    border:1px solid #fecaca;
    font-weight:700;
}

@keyframes spinDtInline {
    to { transform: rotate(360deg); }
}
</style>
"""


@dataclass
class SubtitleItem:
    index: str
    timecode: str
    text: str
    translated_text: str = ""


# =========================
# XỬ LÝ SRT
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
        output.write(f"{item.index}\n{item.timecode}\n{text}\n\n")
    return output.getvalue()


def detect_language(text: str) -> str:
    sample = (text or "").strip()
    if not sample:
        return "unknown"

    if re.search(r"[\u4e00-\u9fff]", sample):
        return "zh"
    if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", sample):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", sample):
        return "ko"
    if re.search(r"[\u0e00-\u0e7f]", sample):
        return "th"
    if re.search(r"[\u0400-\u04FF]", sample):
        return "ru"
    if re.search(r"[\u0600-\u06FF]", sample):
        return "ar"
    if re.search(r"[À-ỹà-ỹĂăÂâĐđÊêÔôƠơƯư]", sample):
        return "vi"

    latin_letters = re.findall(r"[A-Za-z]", sample)
    if latin_letters:
        lower = sample.lower()
        if any(ch in lower for ch in ["¿", "¡", "ñ", "á", "é", "í", "ó", "ú"]):
            return "es"
        if any(ch in lower for ch in ["à", "â", "ç", "è", "é", "ê", "ë", "î", "ï", "ô", "ù", "û", "ü", "œ"]):
            return "fr"
        if any(ch in lower for ch in ["ä", "ö", "ü", "ß"]):
            return "de"
        if any(ch in lower for ch in ["ã", "õ", "ç", "á", "é", "í", "ó", "ú", "â", "ê", "ô"]):
            return "pt"
        return "en_or_latin"

    return "unknown"


def detect_dominant_language(items: List[SubtitleItem], sample_size: int = 80) -> str:
    counts: Dict[str, int] = {}
    checked = 0

    for item in items:
        if checked >= sample_size:
            break
        text = (item.text or "").strip()
        if not text:
            continue
        lang = detect_language(text)
        counts[lang] = counts.get(lang, 0) + 1
        checked += 1

    if not counts:
        return "unknown"

    return max(counts, key=counts.get)


def is_meaningful_text(text: str) -> bool:
    sample = (text or "").strip()
    if not sample:
        return False
    cleaned = re.sub(r"[\W_]+", "", sample, flags=re.UNICODE)
    return bool(cleaned)


def prepare_items_from_source(items: List[SubtitleItem]) -> Tuple[List[SubtitleItem], int]:
    skipped = 0
    for item in items:
        if not is_meaningful_text(item.text):
            item.translated_text = item.text
            skipped += 1
    return items, skipped


def merge_partial_translation(source_items: List[SubtitleItem], partial_items: List[SubtitleItem]) -> Tuple[List[SubtitleItem], int]:
    merged = 0
    limit = min(len(source_items), len(partial_items))

    for i in range(limit):
        partial_text = partial_items[i].text.strip()
        if partial_text:
            source_items[i].translated_text = partial_text
            merged += 1

    return source_items, merged


def build_batches(items: List[SubtitleItem], batch_size: int) -> List[List[SubtitleItem]]:
    pending = [item for item in items if not item.translated_text.strip() and is_meaningful_text(item.text)]
    return [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]


# =========================
# API KEY / GEMINI
# =========================
def create_client(api_key: str):
    return genai.Client(api_key=api_key.strip())


def mask_api_key(api_key: str) -> str:
    api_key = (api_key or "").strip()
    if len(api_key) <= 10:
        return api_key
    return f"{api_key[:6]}...{api_key[-4:]}"


def test_single_api_key(api_key: str, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    api_key = (api_key or "").strip()
    if not api_key:
        return {
            "key": api_key,
            "ok": False,
            "status": "KEY RỖNG",
            "color": "red",
            "detail": "Không có nội dung."
        }

    try:
        client = create_client(api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Chỉ trả lời đúng từ OK",
            config=types.GenerateContentConfig(temperature=0),
        )
        text = (response.text or "").strip().lower()

        if text:
            return {
                "key": api_key,
                "ok": True,
                "status": "KEY OK",
                "color": "green",
                "detail": "Dùng được."
            }

        return {
            "key": api_key,
            "ok": False,
            "status": "PHẢN HỒI RỖNG",
            "color": "red",
            "detail": "Model không trả dữ liệu."
        }

    except Exception as e:
        err = str(e).lower()

        if "quota" in err or "429" in err or "resource_exhausted" in err:
            status = "LIMIT / HẾT QUOTA"
        elif "api key not valid" in err or "invalid" in err or "permission" in err or "unauthenticated" in err:
            status = "KEY KHÔNG HỢP LỆ"
        elif "deadline" in err or "timeout" in err:
            status = "TIMEOUT"
        else:
            status = "LỖI"

        return {
            "key": api_key,
            "ok": False,
            "status": status,
            "color": "red",
            "detail": str(e)
        }


def test_all_api_keys(api_keys: List[str], model_name: str) -> List[Dict[str, Any]]:
    return [test_single_api_key(key, model_name) for key in api_keys]


def build_prompt(batch: List[SubtitleItem], style_prompt: str, source_language: str, target_language: str) -> str:
    rows = []
    for i, item in enumerate(batch, start=1):
        rows.append(f"[{i}] {item.text.replace(chr(13), '').strip()}")
    joined_rows = "\n".join(rows)

    extra = ""
    if style_prompt.strip():
        extra = f"\nYÊU CẦU PHONG CÁCH DỊCH RIÊNG:\n{style_prompt.strip()}\n"

    # THÊM: Xử lý ngôn ngữ đích
    final_target = target_language
    if target_language == "Giống ngôn ngữ nguồn (Sửa lỗi)":
        final_target = "ngôn ngữ gốc (chỉ sửa lỗi chính tả và tối ưu văn phong)"

    if source_language == "Tự động":
        lang_instruction = (
            "- Tự động phát hiện ngôn ngữ của từng dòng.\n"
            "- Nếu file có nhiều ngôn ngữ khác nhau thì vẫn phải xử lý đúng từng dòng.\n"
        )
    else:
        lang_instruction = (
            f"- Xem toàn bộ nội dung đầu vào là {source_language}.\n"
            f"- Ưu tiên dịch theo đặc điểm ngôn ngữ của {source_language}.\n"
        )

    prompt = (
        f"{BASE_SYSTEM_PROMPT}\n"
        f"{extra}\n"
        "NHIỆM VỤ:\n"
        f"{lang_instruction}"
        f"- Dịch tất cả sang **{final_target}**.\n"
        f"- Nếu một dòng đã là {final_target} thì giữ nguyên hoặc chỉnh nhẹ cho tự nhiên hơn.\n\n"
        "ĐỊNH DẠNG TRẢ VỀ:\n"
        "- Mỗi mục phải trả về đúng 1 dòng theo định dạng:\n"
        "[số] bản_dịch\n"
        "- Không được trả về thêm bất kỳ dòng giải thích nào.\n\n"
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
            mapping[int(match.group(1))] = match.group(2).strip()

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
    source_language: str,
    target_language: str
) -> List[str]:
    client = create_client(api_key)
    prompt = build_prompt(batch, style_prompt, source_language, target_language)
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
            time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(str(last_error) if last_error else "Unknown error")


def translate_batch_with_failover(
    batch_id: int,
    batch: List[SubtitleItem],
    worker_slots: List[str],
    model_name: str,
    style_prompt: str,
    source_language: str,
    target_language: str
):
    last_error = ""
    for api_key in worker_slots:
        try:
            translated = try_translate_batch_with_key(
                api_key,
                model_name,
                batch,
                style_prompt,
                source_language,
                target_language
            )
            return batch_id, True, translated, ""
        except Exception as e:
            last_error = str(e)

    return batch_id, False, [item.text for item in batch], last_error


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


# =========================
# XỬ LÝ FILE - TĂNG TỐC
# =========================
def process_one_file(
    file_name: str,
    source_bytes: bytes,
    partial_bytes: Optional[bytes],
    worker_slots: List[str],
    model_name: str,
    style_prompt: str,
    batch_size: int,
    source_language: str,
    target_language: str,
    progress_callback=None
):
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            source_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "file_name": file_name,
                "success": False,
                "error": "Không đọc được file. Hãy lưu SRT dưới dạng UTF-8.",
                "output_bytes": b"",
                "stats": {"total": 0, "skip": 0, "need": 0, "done": 0, "failed_batches": 0, "speed": 0.0},
                "logs": [f"✗ {file_name} | Lỗi decode UTF-8"],
                "detected_lang": "Không xác định",
            }

    source_items = read_srt_content(source_text)
    if not source_items:
        return {
            "file_name": file_name,
            "success": False,
            "error": "File SRT rỗng hoặc không đọc được.",
            "output_bytes": b"",
            "stats": {"total": 0, "skip": 0, "need": 0, "done": 0, "failed_batches": 0, "speed": 0.0},
            "logs": [f"✗ {file_name} | File rỗng hoặc sai định dạng"],
            "detected_lang": "Không xác định",
        }

    detected_lang = detect_dominant_language(source_items)
    detected_lang_label = LANGUAGE_LABELS.get(detected_lang, detected_lang)

    source_items, skipped = prepare_items_from_source(source_items)

    resumed = 0
    if partial_bytes is not None:
        try:
            partial_text = partial_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                partial_text = partial_bytes.decode("utf-8")
            except UnicodeDecodeError:
                partial_text = ""

        if partial_text:
            partial_items = read_srt_content(partial_text)
            source_items, resumed = merge_partial_translation(source_items, partial_items)

    batches = build_batches(source_items, batch_size)
    total_need = sum(len(batch) for batch in batches)

    logs: List[str] = [f"🌐 {file_name} | Phát hiện ngôn ngữ chính: {detected_lang_label}"]
    failed_batches = 0
    done_lines = 0
    start_time = time.time()

    if not batches:
        return {
            "file_name": file_name,
            "success": True,
            "error": "",
            "output_bytes": write_srt_content(source_items).encode("utf-8"),
            "stats": {
                "total": len(source_items),
                "skip": skipped,
                "need": total_need,
                "done": resumed,
                "failed_batches": 0,
                "speed": 0.0
            },
            "logs": logs + ["Không còn dòng nào cần dịch."],
            "detected_lang": detected_lang_label,
        }

    max_parallel_batches = min(MAX_PARALLEL_BATCHES, max(1, len(worker_slots), len(batches)))

    with ThreadPoolExecutor(max_workers=max_parallel_batches) as executor:
        future_map = {}

        for batch_id, batch in enumerate(batches):
            batch_preview = [f"[{item.index}] {item.text[:80]}" for item in batch[:3]]

            if progress_callback:
                progress_callback(
                    event="batch_start",
                    file_name=file_name,
                    batch_id=batch_id,
                    batch_total=len(batches),
                    batch_lines=batch_preview
                )

            future = executor.submit(
                translate_batch_with_failover,
                batch_id,
                batch,
                worker_slots,
                model_name,
                style_prompt,
                source_language,
                target_language
            )
            future_map[future] = (batch_id, batch)

        for future in as_completed(future_map):
            batch_id, batch = future_map[future]
            _, ok, translated_lines, error_text = future.result()

            for item, translated in zip(batch, translated_lines):
                item.translated_text = translated

            done_lines += len(batch)
            elapsed = max(time.time() - start_time, 0.001)
            speed = done_lines / elapsed

            if ok:
                logs.append(f"✓ {file_name} | Batch {batch_id + 1}/{len(batches)} | {len(batch)} dòng | {speed:.1f} dòng/s")
                if progress_callback:
                    progress_callback(
                        event="batch_done",
                        file_name=file_name,
                        batch_id=batch_id,
                        batch_total=len(batches),
                        batch_lines=[f"[{item.index}] {item.translated_text[:80]}" for item in batch[:3]]
                    )
            else:
                failed_batches += 1
                logs.append(f"✗ {file_name} | Batch {batch_id + 1}/{len(batches)} lỗi: {error_text}")
                if progress_callback:
                    progress_callback(
                        event="batch_error",
                        file_name=file_name,
                        batch_id=batch_id,
                        batch_total=len(batches),
                        batch_lines=[f"[{item.index}] {item.text[:80]}" for item in batch[:3]],
                        error_text=error_text
                    )

    final_speed = done_lines / max(time.time() - start_time, 0.001)

    return {
        "file_name": file_name,
        "success": True,
        "error": "",
        "output_bytes": write_srt_content(source_items).encode("utf-8"),
        "stats": {
            "total": len(source_items),
            "skip": skipped + resumed,
            "need": total_need,
            "done": done_lines,
            "failed_batches": failed_batches,
            "speed": final_speed
        },
        "logs": logs,
        "detected_lang": detected_lang_label,
    }


def build_partial_map(partial_upload) -> Dict[str, bytes]:
    result: Dict[str, bytes] = {}
    if partial_upload is None:
        return result

    if partial_upload.name.lower().endswith(".zip"):
        zf = zipfile.ZipFile(io.BytesIO(partial_upload.read()))
        for name in zf.namelist():
            if name.lower().endswith(".srt"):
                result[name.split("/")[-1]] = zf.read(name)

    elif partial_upload.name.lower().endswith(".srt"):
        result[partial_upload.name] = partial_upload.read()

    return result


# =========================
# SESSION STATE
# =========================
def init_state():
    defaults = {
        "zip_bytes": b"",
        "single_bytes": b"",
        "result_ready": False,
        "run_logs": [],
        "finished": False,
        "had_error": False,
        "detected_lang_text": "Chưa phát hiện",
        "api_test_results": [],
        "show_done_popup": False,
        "show_translate_status": False,
        "status_text": "",
        "live_pending_lines": [],
        "live_done_lines": [],
        "stats": {
            "files": 0,
            "total": 0,
            "skip": 0,
            "need": 0,
            "done": 0,
            "failed_batches": 0
        },
        "speed_text": "0 dòng/s",
        "progress_percent": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_run_state():
    st.session_state["zip_bytes"] = b""
    st.session_state["single_bytes"] = b""
    st.session_state["result_ready"] = False
    st.session_state["run_logs"] = []
    st.session_state["finished"] = False
    st.session_state["had_error"] = False
    st.session_state["detected_lang_text"] = "Chưa phát hiện"
    st.session_state["show_done_popup"] = False
    st.session_state["show_translate_status"] = False
    st.session_state["status_text"] = ""
    st.session_state["live_pending_lines"] = []
    st.session_state["live_done_lines"] = []
    st.session_state["stats"] = {"files": 0, "total": 0, "skip": 0, "need": 0, "done": 0, "failed_batches": 0}
    st.session_state["speed_text"] = "0 dòng/s"
    st.session_state["progress_percent"] = 0


def render_translate_status(placeholder):
    if st.session_state.get("show_translate_status"):
        status_text = st.session_state.get("status_text", "Đang dịch phụ đề...")
        placeholder.markdown(
            f"""
            <div class="translate-inline">
                <div class="translate-inline-spinner"></div>
                <div>
                    <div class="translate-inline-title">Đang dịch phụ đề</div>
                    <div class="translate-inline-text">{status_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        placeholder.empty()


# =========================
# UI
# =========================
init_state()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="topbar">'
    '  <div class="brand-wrap">'
    '    <div class="brand-icon">DT</div>'
    '    <div><div class="brand-title">Đình Thái</div><div class="brand-sub">SRT Translator Studio</div></div>'
    '  </div>'
    '  <div class="version-pill">V11 FAST MODE</div>'
    '</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, 1.58], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-title">📤 Upload Nhiều File SRT</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Chọn 1 hoặc nhiều file .srt", type=["srt"], accept_multiple_files=True)
    partial_zip = st.file_uploader("File dịch dở (ZIP hoặc 1 SRT cùng tên để dịch tiếp)", type=["zip", "srt"])
    st.markdown(
        '<div class="card-note">Đã bỏ phần preview để tăng tốc độ. Bản này ưu tiên dịch nhanh hơn.</div></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="card"><div class="card-title">⚙️ Cấu Hình Dịch</div>', unsafe_allow_html=True)
    # THÊM: Điều chỉnh chia cột để có chỗ cho Ngôn ngữ Đích
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        model_name = st.selectbox("MODEL", options=["gemini-2.0-flash", "gemini-1.5-flash"], index=0)
    with c2:
        batch_size = st.number_input("BATCH SIZE", min_value=1, max_value=200, value=DEFAULT_BATCH_SIZE, step=1)
    with c3:
        source_language = st.selectbox("NGÔN NGỮ NGUỒN", options=SOURCE_LANGUAGE_OPTIONS, index=0)
    # THÊM: Lựa chọn ngôn ngữ đích
    with c4:
        target_language = st.selectbox("NGÔN NGỮ ĐÍCH", options=TARGET_LANGUAGE_OPTIONS, index=0)

    output_zip_name = st.text_input("TÊN FILE ZIP XUẤT", value="srt_translated_bundle.zip")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">✍️ Prompt Phong Cách Dịch</div>', unsafe_allow_html=True)
    style_prompt = st.text_area(
        "Prompt",
        value="Dịch tự nhiên, mượt như phụ đề phim. Xưng hô phù hợp ngữ cảnh, ưu tiên câu ngắn gọn, dễ đọc.",
        height=130,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🔑 API Key</div>', unsafe_allow_html=True)
    keys_text = st.text_area(
        "API Keys",
        placeholder="AIza...\nAIza...\nAIza...",
        height=150,
        label_visibility="collapsed"
    )

    batch_text = st.text_area(
        "Slots",
        value="3\n3\n3",
        height=92,
        help="Mỗi dòng là số slot tương ứng với từng API key"
    )

    t1, t2 = st.columns([1, 1])
    with t1:
        test_key_btn = st.button("🧪 Test API Key", use_container_width=True)
    with t2:
        clear_key_test_btn = st.button("♻️ Xóa kết quả test", use_container_width=True)

    if st.session_state["api_test_results"]:
        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
        for item in st.session_state["api_test_results"]:
            color_class = "key-yellow"
            if item["color"] == "green":
                color_class = "key-green"
            elif item["color"] == "red":
                color_class = "key-red"

            st.markdown(
                f'''
                <div class="key-box">
                    <div class="key-row">
                        <div class="key-name">{mask_api_key(item["key"])}</div>
                        <div class="key-status {color_class}">{item["status"]}</div>
                    </div>
                    <div class="key-detail">{item["detail"]}</div>
                </div>
                ''',
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🚀 Điều Khiển</div>', unsafe_allow_html=True)

    translate_status_placeholder = st.empty()
    render_translate_status(translate_status_placeholder)

    progress_placeholder = st.empty()
    percent_placeholder = st.empty()

    b1, b2 = st.columns([5, 1])
    with b1:
        run_btn = st.button("▶ Bắt Đầu Dịch Nhiều File", type="primary", use_container_width=True)
    with b2:
        clear_btn = st.button("🗑", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    stats = st.session_state["stats"]
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Số file</div><div class="metric-value">{stats["files"]}</div><div class="metric-sub">Upload cùng lúc</div></div>',
            unsafe_allow_html=True
        )
    with g2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Tổng subtitle</div><div class="metric-value">{stats["total"]}</div><div class="metric-sub">Tất cả file</div></div>',
            unsafe_allow_html=True
        )
    with g3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Tốc độ</div><div class="metric-value">{st.session_state["speed_text"]}</div><div class="metric-sub">Subtitle / giây</div></div>',
            unsafe_allow_html=True
        )

    g4, g5, g6 = st.columns(3)
    with g4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Cần dịch</div><div class="metric-value">{stats["need"]}</div><div class="metric-sub">Dòng có nội dung</div></div>',
            unsafe_allow_html=True
        )
    with g5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Đã xong</div><div class="metric-value">{stats["done"]}</div><div class="metric-sub">Trong lượt này</div></div>',
            unsafe_allow_html=True
        )
    with g6:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Batch lỗi</div><div class="metric-value">{stats["failed_batches"]}</div><div class="metric-sub">Cần chạy lại nếu có</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="card"><div class="card-title">🖥 Console Logs</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()
    if st.session_state["run_logs"]:
        log_placeholder.code("\n".join(st.session_state["run_logs"][-14:]))
    else:
        log_placeholder.info("Đang chờ upload file và bấm bắt đầu dịch.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📡 Tiến Trình Dịch</div>', unsafe_allow_html=True)
    live_c1, live_c2 = st.columns(2)

    with live_c1:
        st.markdown('<div class="small">Đang dịch</div>', unsafe_allow_html=True)
        pending_live_placeholder = st.empty()
        if st.session_state["live_pending_lines"]:
            pending_html = "".join([f'<div class="batch-line-pending">{x}</div>' for x in st.session_state["live_pending_lines"][-6:]])
            pending_live_placeholder.markdown(f'<div class="batch-live-box">{pending_html}</div>', unsafe_allow_html=True)
        else:
            pending_live_placeholder.markdown('<div class="batch-live-box"><div class="small">Chưa có dòng nào đang dịch.</div></div>', unsafe_allow_html=True)

    with live_c2:
        st.markdown('<div class="small">Đã hoàn thành</div>', unsafe_allow_html=True)
        done_live_placeholder = st.empty()
        if st.session_state["live_done_lines"]:
            done_html = "".join([f'<div class="batch-line-done">{x}</div>' for x in st.session_state["live_done_lines"][-6:]])
            done_live_placeholder.markdown(f'<div class="batch-live-box">{done_html}</div>', unsafe_allow_html=True)
        else:
            done_live_placeholder.markdown('<div class="batch-live-box"><div class="small">Chưa có dòng nào hoàn thành.</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# TEST KEY
# =========================
if clear_key_test_btn:
    st.session_state["api_test_results"] = []
    st.rerun()

if test_key_btn:
    raw_keys = [line.strip() for line in keys_text.splitlines() if line.strip()]
    if not raw_keys:
        st.warning("Bạn chưa nhập API key để test.")
    else:
        with st.spinner("Đang test API key..."):
            st.session_state["api_test_results"] = test_all_api_keys(raw_keys, model_name)
        st.rerun()


# =========================
# CLEAR
# =========================
if clear_btn:
    reset_run_state()
    st.rerun()


# =========================
# RUN
# =========================
if run_btn:
    reset_run_state()

    if not uploaded_files:
        st.error("Bạn chưa upload file SRT nào.")
    else:
        api_keys, worker_slots = collect_api_keys_and_slots(keys_text, batch_text)

        if st.session_state["api_test_results"]:
            ok_keys = {item["key"].strip() for item in st.session_state["api_test_results"] if item.get("ok")}
            if ok_keys:
                api_keys = [k for k in api_keys if k.strip() in ok_keys]
                worker_slots = [k for k in worker_slots if k.strip() in ok_keys]

        if not api_keys:
            st.error("Bạn chưa nhập API key nào hoặc tất cả key đã bị loại sau khi test.")
        elif not worker_slots:
            st.error("Không có worker slot hợp lệ.")
        else:
            partial_map = build_partial_map(partial_zip)
            progress_bar = progress_placeholder.progress(0.0)
            percent_placeholder.markdown("**0%**")

            logs: List[str] = []
            start_time = time.time()
            results: List[Dict] = []

            total_files = len(uploaded_files)
            st.session_state["stats"]["files"] = total_files

            pending_box = pending_live_placeholder
            done_box = done_live_placeholder

            def update_live_boxes():
                if st.session_state["live_pending_lines"]:
                    pending_html = "".join(
                        [f'<div class="batch-line-pending">{x}</div>' for x in st.session_state["live_pending_lines"][-6:]]
                    )
                    pending_box.markdown(f'<div class="batch-live-box">{pending_html}</div>', unsafe_allow_html=True)
                else:
                    pending_box.markdown('<div class="batch-live-box"><div class="small">Chưa có dòng nào đang dịch.</div></div>', unsafe_allow_html=True)

                if st.session_state["live_done_lines"]:
                    done_html = "".join(
                        [f'<div class="batch-line-done">{x}</div>' for x in st.session_state["live_done_lines"][-6:]]
                    )
                    done_box.markdown(f'<div class="batch-live-box">{done_html}</div>', unsafe_allow_html=True)
                else:
                    done_box.markdown('<div class="batch-live-box"><div class="small">Chưa có dòng nào hoàn thành.</div></div>', unsafe_allow_html=True)

            def ui_progress_callback(event, file_name, batch_id, batch_total, batch_lines, error_text=""):
                if event == "batch_start":
                    st.session_state["show_translate_status"] = True
                    st.session_state["status_text"] = f"{file_name} • Batch {batch_id + 1}/{batch_total}"
                    st.session_state["live_pending_lines"] = [f"{file_name} • {line}" for line in batch_lines[:3]]
                    update_live_boxes()
                    render_translate_status(translate_status_placeholder)

                elif event == "batch_done":
                    st.session_state["live_done_lines"].extend([f"{file_name} • {line}" for line in batch_lines[:3]])
                    st.session_state["live_pending_lines"] = []
                    update_live_boxes()

                elif event == "batch_error":
                    st.session_state["live_done_lines"].append(f"{file_name} • Batch {batch_id + 1} lỗi: {error_text}")
                    st.session_state["live_pending_lines"] = []
                    update_live_boxes()

            completed_files = 0
            agg_total = 0
            agg_skip = 0
            agg_need = 0
            agg_done = 0
            agg_failed = 0

            for f in uploaded_files:
                st.session_state["show_translate_status"] = True
                st.session_state["status_text"] = f"Đang chuẩn bị dịch file {f.name}"
                render_translate_status(translate_status_placeholder)

                # TRUYỀN THÊM target_language vào hàm xử lý file
                result = process_one_file(
                    f.name,
                    f.read(),
                    partial_map.get(f.name),
                    worker_slots,
                    model_name,
                    style_prompt,
                    int(batch_size),
                    source_language,
                    target_language,
                    progress_callback=ui_progress_callback
                )

                results.append(result)
                completed_files += 1

                agg_total += result["stats"]["total"]
                agg_skip += result["stats"]["skip"]
                agg_need += result["stats"]["need"]
                agg_done += result["stats"]["done"]
                agg_failed += result["stats"]["failed_batches"]

                elapsed = max(time.time() - start_time, 0.001)
                speed = agg_done / elapsed if agg_done > 0 else 0.0
                percent = int((completed_files / total_files) * 100)

                st.session_state["stats"] = {
                    "files": total_files,
                    "total": agg_total,
                    "skip": agg_skip,
                    "need": agg_need,
                    "done": agg_done,
                    "failed_batches": agg_failed,
                }
                st.session_state["speed_text"] = f"{speed:.1f} dòng/s"
                st.session_state["progress_percent"] = percent
                st.session_state["detected_lang_text"] = result.get("detected_lang", "Không xác định")

                progress_bar.progress(completed_files / total_files)
                percent_placeholder.markdown(f"**{percent}%**")

                logs.extend(result["logs"][-4:])
                if result["error"]:
                    logs.append(f"✗ {result['file_name']}: {result['error']}")
                    st.session_state["had_error"] = True

                st.session_state["run_logs"] = logs
                log_placeholder.code("\n".join(logs[-14:]))

            st.session_state["show_translate_status"] = False
            st.session_state["status_text"] = ""
            render_translate_status(translate_status_placeholder)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for result in results:
                    if result["output_bytes"]:
                        zf.writestr(result["file_name"], result["output_bytes"])

            st.session_state["zip_bytes"] = zip_buffer.getvalue()
            if len(results) == 1:
                st.session_state["single_bytes"] = results[0]["output_bytes"]

            st.session_state["result_ready"] = True
            st.session_state["finished"] = True
            st.session_state["show_done_popup"] = True

            if st.session_state["stats"]["failed_batches"] > 0:
                st.session_state["had_error"] = True


# =========================
# THÔNG BÁO KẾT QUẢ
# =========================
if st.session_state["finished"]:
    if st.session_state["had_error"]:
        st.markdown(
            '<div class="status-warn">Đã xử lý xong phần thành công. Một số batch hoặc file còn lỗi.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-ok">Đã xử lý xong toàn bộ sang ' + target_language + '.</div>',
            unsafe_allow_html=True
        )

if st.session_state["result_ready"] and st.session_state["zip_bytes"]:
    st.download_button(
        label="Tải ZIP kết quả",
        data=st.session_state["zip_bytes"],
        file_name=output_zip_name,
        mime="application/zip",
        use_container_width=True,
    )

# =========================
# POPUP HOÀN THÀNH
# =========================
if st.session_state["show_done_popup"]:
    popup_col1, popup_col2 = st.columns([10, 1])
    with popup_col2:
        if st.button("✕", key="close_done_popup"):
            st.session_state["show_done_popup"] = False
            st.rerun()

    st.markdown(
        f"""
        <div class="done-popup">
            <div class="done-popup-body">
                <div class="done-check">✓</div>
                <div class="done-title">Đã dịch xong</div>
                <div class="done-text">
                    Mục tiêu: <b>{target_language}</b><br>
                    Số file: <b>{st.session_state["stats"]["files"]}</b><br>
                    Tổng dòng đã xử lý: <b>{st.session_state["stats"]["done"]}</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
