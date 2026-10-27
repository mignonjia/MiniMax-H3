"""Global and per-chunk captions in MiniMax-H3's prompt register.

A clip arrives as a structured caption: global fields (style, lighting, view,
subjects, objects, first-frame description, ...) plus timed events. The global
caption is the global fields only (no events: a chunk must not read the
future); each chunk's caption is the event covering the chunk's centre. Every
absolute time is dropped from chunk captions: it names the chunk's position in
the clip, which stops meaning anything once a rollout runs past the trained
length (PAN own-caption lesson, `drop_chunk_event_keys`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from h3_causal.dataset import geometry as geo

TIME_KEYS = frozenset({"timestamp", "time_range_sec", "time_range", "start", "end", "event_index"})
SKIP_GLOBAL_KEYS = frozenset({"events", "quality", "hud_persistent", "h3_prompt"})
_SUBJECT_TAG = re.compile(r"<\|?[A-Za-z]+\d*\|?>")


@dataclass
class Event:
    start: float
    end: float
    fields: dict


@dataclass
class ClipCaptions:
    global_text: str
    chunk_texts: list[str]  # unique captions
    chunk_index: list[int]  # per chunk, into chunk_texts (-1 = no caption)


def parse_time_range(value) -> tuple[float, float]:
    """'4.3 - 9.4', '1 to 2', [a, b] -> (a, b) seconds."""
    if isinstance(value, (list, tuple)):
        return float(value[0]), float(value[1])
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value).replace(" - ", " to "))
    if len(numbers) < 2:
        raise ValueError(f"cannot parse time range {value!r}")
    return float(numbers[0]), float(numbers[1])


def _subject_names(caption: dict) -> dict[str, str]:
    names = {}
    for subject in caption.get("subjects_lifecycle") or []:
        if isinstance(subject, dict) and subject.get("subject_id") and subject.get("display_name"):
            names[subject["subject_id"]] = subject["display_name"]
    return names


def _replace_tags(text: str, names: dict[str, str]) -> str:
    for tag, name in names.items():
        text = text.replace(tag, name)
    return _SUBJECT_TAG.sub("the subject", text)


def _strip_times(value):
    if isinstance(value, dict):
        return {k: _strip_times(v) for k, v in value.items() if k not in TIME_KEYS}
    if isinstance(value, list):
        return [_strip_times(v) for v in value]
    return value


def _render(value) -> str:
    """Compact prose for nested caption fields."""
    if isinstance(value, dict):
        parts = [_render(v) for v in value.values()]
        return "; ".join(p for p in parts if p)
    if isinstance(value, list):
        parts = [_render(v) for v in value]
        return ", ".join(p for p in parts if p)
    if value is None:
        return ""
    return str(value).strip()


def _label(key: str) -> str:
    return key.replace("_", " ")


def _sentence(key: str, text: str) -> str:
    return f"{_label(key).capitalize()}: {text.rstrip('. ')}."


def global_caption_text(caption: dict) -> str:
    """`integrated_multimodal_description:` built from the non-event fields, or `global_text` verbatim."""
    if caption.get("global_text"):
        return caption["global_text"]
    names = _subject_names(caption)
    parts = []
    for key in ("video_style", "perspective_view", "lighting"):
        if caption.get(key):
            parts.append(_sentence(key, _render(caption[key])))
    subjects = []
    for subject in caption.get("subjects_lifecycle") or []:
        if isinstance(subject, dict):
            subjects.append(f"{subject.get('display_name', 'subject')} ({_render(subject.get('subject_attribute'))})")
    if subjects:
        parts.append(_sentence("subjects", "; ".join(subjects)))
    if caption.get("objects"):
        parts.append(_sentence("objects", _render(caption["objects"])))
    if caption.get("init_frame_visual_description"):
        parts.append(_render(caption["init_frame_visual_description"]))
    for key, value in caption.items():
        if key in SKIP_GLOBAL_KEYS or key in TIME_KEYS or key in {
            "video_style", "perspective_view", "lighting", "subjects_lifecycle", "objects", "init_frame_visual_description"
        }:
            continue
        rendered = _render(_strip_times(value))
        if rendered:
            parts.append(_sentence(key, rendered))
    return "integrated_multimodal_description: " + _replace_tags(" ".join(parts), names)


def event_caption_text(event: Event, names: dict[str, str]) -> str:
    fields = _strip_times(event.fields)
    parts = []
    if fields.get("event_description"):
        parts.append(_render(fields.pop("event_description")))
    for key, value in fields.items():
        rendered = _render(value)
        if rendered:
            parts.append(_sentence(key, rendered))
    return _replace_tags(" ".join(parts), names)


def loki_events(caption: dict) -> list[Event]:
    return [Event(*parse_time_range(e["timestamp"]), e) for e in caption.get("events") or []]


def game_v7_events(chunk_captions: list) -> list[Event]:
    """v7 `chunk_captions`: one JSON list of per-second events per second of the clip."""
    events = []
    for entry in chunk_captions:
        items = json.loads(entry) if isinstance(entry, str) else entry
        for item in items:
            events.append(Event(*parse_time_range(item["time_range_sec"]), item))
    return events


def build_clip_captions(caption: dict, events: list[Event], num_chunks: int) -> ClipCaptions:
    names = _subject_names(caption)
    unique: dict[str, int] = {}
    chunk_index = []
    for chunk in range(num_chunks):
        start, end = geo.chunk_time_range_sec(chunk)
        centre = 0.5 * (start + end)
        covering = [e for e in events if e.start <= centre < e.end]
        if not covering and events:
            # Gaps between annotated events: the nearest event by centre distance.
            covering = [min(events, key=lambda e: min(abs(e.start - centre), abs(e.end - centre)))]
        if not covering:
            chunk_index.append(-1)
            continue
        text = event_caption_text(covering[0], names)
        if text not in unique:
            unique[text] = len(unique)
        chunk_index.append(unique[text])
    return ClipCaptions(global_caption_text(caption), list(unique), chunk_index)
