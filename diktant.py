#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор аудио-диктанта из текста с заданным темпом (слов в минуту).

Идея темпа: WPM здесь — это общий темп диктанта (слова / минуты всего аудио),
а не скорость речи диктора. Речь озвучивается в естественном темпе, а нужный
WPM набирается за счёт пауз между смысловыми отрезками — как на настоящем
диктанте, когда учитель ждёт, пока класс допишет.

Движки синтеза:
  * yandex — Yandex SpeechKit v1 (лучшее качество для русского, нужен API-ключ)
  * sapi   — офлайн, встроенный синтез Windows (System.Speech), без ключей

Зависимостей нет — только стандартная библиотека.

Примеры:
    python diktant.py text.txt -o diktant.wav --wpm 55
    python diktant.py text.txt -o diktant.wav --wpm 45 --repeat 2 --full-read before
    python diktant.py text.txt --dry-run --wpm 60
    python diktant.py text.txt -o d.wav --backend sapi --list-voices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Формат аудио-конвейера: везде 48 кГц, 16 бит, моно — тогда куски склеиваются
# побайтово, без ресемплинга и без ffmpeg.
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2
CHANNELS = 1
BYTES_PER_SEC = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS


# ===========================================================================
# Разбор текста на смысловые отрезки
# ===========================================================================

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+(?:[-'’][А-Яа-яЁёA-Za-z0-9]+)*")

# Сокращения, после точки которых предложение не заканчивается.
ABBREVIATIONS = {
    "т", "е", "д", "п", "к", "н", "о", "г", "гг", "в", "вв", "им", "ул", "пр",
    "др", "пр-т", "проф", "акад", "доц", "стр", "рис", "табл", "см", "ср",
    "кв", "руб", "коп", "мин", "сек", "ч", "мл", "тыс", "млн", "млрд", "обл",
    "р", "с", "тов", "гр", "изд", "сост", "яз",
}

SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?…])["»”\')\]]*\s+')
# Места, где диктор естественно делает вдох внутри длинного предложения.
CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:—–])\s+")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _ends_with_abbreviation(chunk: str) -> bool:
    m = re.search(r"([А-Яа-яЁёA-Za-z-]+)\.$", chunk.strip())
    return bool(m) and m.group(1).lower() in ABBREVIATIONS


NUMBERING_RE = re.compile(r"^\(?\d{1,3}[.)]$")  # маркер нумерованного списка: "1.", "12)"
# Инициалы: «Б.», «Л.» — одна заглавная буква с точкой в конце уже
# накопленного куска. Не конец предложения, даже если дальше снова заглавная
# буква (следующий инициал или фамилия): «Б. Л. Пастернак».
INITIAL_RE = re.compile(r"(?:^|\s)[А-ЯЁA-Z]\.$")


def split_sentences(text: str) -> list[str]:
    """Делит текст на предложения, не спотыкаясь о «и т. д.», «в 1812 г.»,
    номера пунктов списка («1. Делу время...»), инициалы («Б. Л. Пастернак»)
    и т. п."""
    pieces = [p for p in SENTENCE_BOUNDARY.split(text) if p.strip()]
    sentences: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if sentences and (
            _ends_with_abbreviation(sentences[-1])
            or NUMBERING_RE.match(sentences[-1])       # "1." — номер пункта списка
            or INITIAL_RE.search(sentences[-1])         # "...Б." — инициал
            or re.match(r"^[а-яёa-z]", piece)           # продолжение со строчной буквы
        ):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _split_by_words(text: str, max_words: int) -> list[str]:
    """Режет текст без знаков препинания на чанки по max_words слов подряд,
    не отрывая инициалы («Б. Л.») от следующего за ними слова — фамилии."""
    words = text.split(" ")
    n = len(words)
    chunks: list[str] = []
    i = 0
    while i < n:
        j = min(i + max_words, n)
        while j < n and INITIAL_RE.search(" " + words[j - 1]):
            j += 1
        chunks.append(" ".join(words[i:j]))
        i = j
    return chunks


def split_long_sentence(sentence: str, max_words: int, min_words: int) -> list[str]:
    """Режет предложение на отрезки по знакам препинания — так, чтобы диктор
    делал паузу не только в конце предложения, но и на каждой запятой, тире,
    двоеточии. Слишком длинный кусок без пунктуации дробится по словам,
    слишком короткий осколок приклеивается к соседнему."""
    fragments = [f.strip() for f in CLAUSE_BOUNDARY.split(sentence) if f.strip()]
    if len(fragments) < 2:
        return (_split_by_words(sentence, max_words)
                if count_words(sentence) > max_words else [sentence])

    parts: list[str] = []
    for frag in fragments:
        if count_words(frag) > max_words:
            parts.extend(_split_by_words(frag, max_words))
        else:
            parts.append(frag)

    merged: list[str] = []
    for part in parts:
        if merged and count_words(part) < min_words:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


@dataclass
class Segment:
    """Один отрезок, который диктор читает целиком, а потом ждёт."""

    text: str
    sentence_index: int
    is_sentence_end: bool
    sentence_text: str = ""  # предложение целиком, до разбивки на отрезки
    words: int = 0
    speech_sec: float = 0.0   # длительность речи (одного прочтения)
    pause_sec: float = 0.0    # пауза после отрезка
    pcm: bytes = field(default=b"", repr=False)

    # «Рекап» — фраза целиком перед тем, как она пойдёт по кусочкам.
    # Ставится на ПЕРВЫЙ отрезок предложения: сперва рекап, потом сам отрезок.
    # Синтезируется ОДНИМ вызовом TTS по sentence_text — не склейкой уже
    # нарезанных отрезков, иначе на месте будущих запятых слышны лишние паузы.
    recap: bool = False
    recap_pre_pause: float = 0.0   # пауза после рекапа, перед началом надиктовки по частям
    recap_speech_sec: float = 0.0  # длительность самого рекапа
    recap_pcm: bytes = field(default=b"", repr=False)


def build_segments(text: str, max_words: int, min_words: int) -> list[Segment]:
    segments: list[Segment] = []
    for si, sentence in enumerate(split_sentences(text)):
        parts = split_long_sentence(sentence, max_words, min_words)
        for pi, part in enumerate(parts):
            segments.append(
                Segment(
                    text=part,
                    sentence_index=si,
                    is_sentence_end=(pi == len(parts) - 1),
                    sentence_text=sentence,
                    words=count_words(part),
                )
            )
    return segments


def sentence_groups(segments: list[Segment]) -> list[list[Segment]]:
    """Группирует отрезки по предложениям, которым они принадлежат."""
    groups: list[list[Segment]] = []
    start = 0
    for idx, seg in enumerate(segments):
        if seg.is_sentence_end:
            groups.append(segments[start:idx + 1])
            start = idx + 1
    return groups


# ===========================================================================
# Движки синтеза речи
# ===========================================================================


class SynthError(RuntimeError):
    pass


class Backend:
    name = "base"

    def synthesize(self, text: str, speed: float) -> bytes:
        """Возвращает сырой PCM 48 кГц / 16 бит / моно."""
        raise NotImplementedError


class YandexBackend(Backend):
    """Yandex SpeechKit v1, формат lpcm — сразу нужный нам PCM."""

    name = "yandex"
    URL = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    MAX_CHARS = 4500  # лимит API — 5000, берём с запасом

    def __init__(self, api_key: str, voice: str, role: str | None,
                 emotion: str | None, folder_id: str | None = None):
        self.api_key = api_key
        self.voice = voice
        self.role = role
        self.emotion = emotion
        self.folder_id = folder_id

    def synthesize(self, text: str, speed: float) -> bytes:
        if len(text) > self.MAX_CHARS:
            raise SynthError(f"Отрезок длиннее {self.MAX_CHARS} символов: {text[:60]}…")

        payload = {
            "text": text,
            "lang": "ru-RU",
            "voice": self.voice,
            "speed": f"{speed:.2f}",
            "format": "lpcm",
            "sampleRateHertz": str(SAMPLE_RATE),
        }
        if self.role:
            payload["role"] = self.role
        if self.emotion:
            payload["emotion"] = self.emotion
        if self.folder_id:
            payload["folderId"] = self.folder_id

        req = urllib.request.Request(
            self.URL,
            data=urllib.parse.urlencode(payload).encode("utf-8"),
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            hint = ""
            if exc.code == 401:
                hint = "\nПроверьте YANDEX_API_KEY (Api-Key сервисного аккаунта)."
            elif exc.code == 403:
                hint = "\nУ сервисного аккаунта нет роли ai.speechkit-tts.user."
            raise SynthError(f"Yandex SpeechKit HTTP {exc.code}: {body}{hint}") from exc
        except urllib.error.URLError as exc:
            raise SynthError(f"Нет связи с Yandex SpeechKit: {exc.reason}") from exc


PS_SYNTH_SCRIPT = r"""
param([string]$TextFile, [string]$OutFile, [string]$VoiceName, [int]$Rate)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($VoiceName) { $synth.SelectVoice($VoiceName) }
$synth.Rate = $Rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    48000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($OutFile, $fmt)
$synth.Speak([System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8))
$synth.SetOutputToNull()
$synth.Dispose()
"""

PS_LIST_VOICES = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.GetInstalledVoices() | ForEach-Object {
    $i = $_.VoiceInfo
    "{0}`t{1}`t{2}" -f $i.Name, $i.Culture.Name, $i.Gender
}
"""


def _run_powershell(script: str, args: list[str] | None = None) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False,
                                     encoding="utf-8-sig") as fh:
        fh.write(script)
        script_path = fh.name
    try:
        cmd = ["powershell", "-NoProfile", "-NonInteractive",
               "-ExecutionPolicy", "Bypass", "-File", script_path]
        cmd += args or []
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise SynthError(f"PowerShell/SAPI: {proc.stderr.strip()}")
        return proc.stdout
    finally:
        os.unlink(script_path)


class SapiBackend(Backend):
    """Офлайн-синтез средствами Windows. Качество зависит от установленных голосов."""

    name = "sapi"

    def __init__(self, voice: str | None):
        if sys.platform != "win32":
            raise SynthError("Движок sapi доступен только в Windows.")
        self.voice = voice

    @staticmethod
    def list_voices() -> list[tuple[str, str, str]]:
        out = _run_powershell(PS_LIST_VOICES)
        voices = []
        for line in out.splitlines():
            if line.strip():
                parts = line.rstrip("\n").split("\t")
                while len(parts) < 3:
                    parts.append("")
                voices.append((parts[0], parts[1], parts[2]))
        return voices

    def synthesize(self, text: str, speed: float) -> bytes:
        # SAPI Rate: -10..10, где 0 — норма. Примерно логарифмическая шкала.
        rate = max(-10, min(10, round((speed - 1.0) * 10)))
        tmpdir = Path(tempfile.mkdtemp(prefix="diktant_"))
        try:
            txt_path = tmpdir / "chunk.txt"
            wav_path = tmpdir / "chunk.wav"
            txt_path.write_text(text, encoding="utf-8")
            _run_powershell(
                PS_SYNTH_SCRIPT,
                ["-TextFile", str(txt_path), "-OutFile", str(wav_path),
                 "-VoiceName", self.voice or "", "-Rate", str(rate)],
            )
            with wave.open(str(wav_path), "rb") as wf:
                if (wf.getframerate(), wf.getsampwidth(), wf.getnchannels()) != (
                    SAMPLE_RATE, SAMPLE_WIDTH, CHANNELS
                ):
                    raise SynthError("SAPI вернул неожиданный формат аудио.")
                return wf.readframes(wf.getnframes())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Кэш синтеза — чтобы не платить за повторные запросы при подборе темпа
# ===========================================================================


class SynthCache:
    def __init__(self, backend: Backend, cache_dir: Path | None):
        self.backend = backend
        self.dir = cache_dir
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, text: str, speed: float) -> str:
        ident = json.dumps(
            [self.backend.name, getattr(self.backend, "voice", ""),
             getattr(self.backend, "role", ""), getattr(self.backend, "emotion", ""),
             round(speed, 3), text],
            ensure_ascii=False,
        )
        return hashlib.sha256(ident.encode("utf-8")).hexdigest()

    def get(self, text: str, speed: float) -> bytes:
        if not self.dir:
            self.misses += 1
            return self.backend.synthesize(text, speed)
        path = self.dir / f"{self._key(text, speed)}.pcm"
        if path.exists():
            self.hits += 1
            return path.read_bytes()
        self.misses += 1
        pcm = self.backend.synthesize(text, speed)
        path.write_bytes(pcm)
        return pcm


# ===========================================================================
# Операции над PCM
# ===========================================================================


def silence(seconds: float) -> bytes:
    frames = max(0, int(round(seconds * SAMPLE_RATE)))
    return b"\x00\x00" * frames


def duration_of(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_SEC


def trim_silence(pcm: bytes, threshold: int = 400, keep_ms: int = 40) -> bytes:
    """Срезает тишину по краям, чтобы паузы получались ровно заданной длины."""
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % SAMPLE_WIDTH)])
    if not samples:
        return pcm
    if sys.byteorder == "big":
        samples.byteswap()

    window = SAMPLE_RATE // 100  # 10 мс
    start, end = 0, len(samples)
    for i in range(0, len(samples), window):
        if max(abs(v) for v in samples[i:i + window]) > threshold:
            start = i
            break
    else:
        return b""
    for i in range(len(samples) - window, -1, -window):
        if max(abs(v) for v in samples[i:i + window]) > threshold:
            end = min(len(samples), i + window)
            break

    keep = int(SAMPLE_RATE * keep_ms / 1000)
    start = max(0, start - keep)
    end = min(len(samples), end + keep)
    return pcm[start * SAMPLE_WIDTH: end * SAMPLE_WIDTH]


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def encode_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "128k") -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SynthError(
            "Для .mp3 нужен ffmpeg в PATH. Либо сохраните в .wav, "
            "либо установите ffmpeg (winget install Gyan.FFmpeg)."
        )
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path),
         "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)],
        check=True,
    )


# ===========================================================================
# Планирование пауз под целевой WPM
# ===========================================================================


@dataclass
class Plan:
    segments: list[Segment]
    lead_in: float
    tail: float
    repeat: int
    repeat_gap: float
    speech_total: float
    pause_total: float
    recap_pre_total: float = 0.0
    recap_speech_total: float = 0.0

    @property
    def total(self) -> float:
        return (self.lead_in + self.speech_total + self.pause_total
                + self.recap_pre_total + self.recap_speech_total + self.tail)

    @property
    def words(self) -> int:
        return sum(s.words for s in self.segments)

    @property
    def actual_wpm(self) -> float:
        return self.words / (self.total / 60) if self.total else 0.0


def plan_pauses(segments: list[Segment], wpm: float, *, repeat: int,
                repeat_gap: float, min_pause: float, sentence_extra: float,
                clause_extra: float, lead_in: float, tail: float) -> tuple[Plan, str | None]:
    """Распределяет паузы так, чтобы весь диктант уложился в целевой темп.

    Пауза пропорциональна числу слов в отрезке: длиннее фраза — дольше ждём.
    В конце предложения добавляется sentence_extra, после запятой/тире/
    двоеточия внутри предложения — clause_extra (короче, просто на вдох).
    Рекап (фраза целиком перед надиктовкой по частям — см. main()) уже
    отмечен на сегментах заранее и учитывается как часть речи наравне с
    самими отрезками: время на него берётся из общего бюджета темпа, а не
    в ущерб паузам на запись.
    """
    words = sum(s.words for s in segments)
    target_total = words / wpm * 60 if wpm > 0 else 0.0

    # Речь: каждый отрезок читается `repeat` раз, между повторами — короткий зазор.
    speech_total = sum(s.speech_sec for s in segments) * repeat
    gaps_total = sum(repeat_gap for _ in segments) * (repeat - 1)

    recap_pre_total = sum(s.recap_pre_pause for s in segments if s.recap)
    recap_speech_total = sum(s.recap_speech_sec for s in segments if s.recap)

    # Добавка за знак препинания в конце отрезка — входит в бюджет пауз.
    extras = [sentence_extra if s.is_sentence_end else clause_extra for s in segments]

    fixed = (lead_in + tail + speech_total + gaps_total
             + recap_pre_total + recap_speech_total)
    budget = target_total - fixed - sum(extras)
    warning = None

    floor_total = min_pause * len(segments)
    if budget < floor_total:
        needed = fixed + sum(extras) + floor_total
        warning = (
            f"Темп {wpm:g} сл/мин недостижим: даже с минимальными паузами "
            f"({min_pause:g} с) диктант займёт {fmt_time(needed)} "
            f"({words / (needed / 60):.1f} сл/мин). "
            f"Ускорьте речь (--speed), уменьшите --min-pause, --clause-extra, "
            f"отключите --no-recap или снизьте --wpm."
        )
        for seg, extra in zip(segments, extras):
            seg.pause_sec = min_pause + extra
    else:
        extra_budget = budget - floor_total
        for seg, extra in zip(segments, extras):
            share = (seg.words / words) if words else 1 / len(segments)
            seg.pause_sec = min_pause + extra_budget * share + extra

    pause_total = sum(s.pause_sec for s in segments) + gaps_total
    plan = Plan(
        segments=segments, lead_in=lead_in, tail=tail, repeat=repeat,
        repeat_gap=repeat_gap, speech_total=speech_total, pause_total=pause_total,
        recap_pre_total=recap_pre_total, recap_speech_total=recap_speech_total,
    )
    return plan, warning


def estimate_speech_sec(text: str, speed: float, natural_wpm: float = 145.0) -> float:
    """Оценка длительности речи без обращения к синтезу (для --dry-run)."""
    return count_words(text) / (natural_wpm * speed) * 60


def render(plan: Plan) -> bytes:
    out = bytearray(silence(plan.lead_in))
    for seg in plan.segments:
        if seg.recap and seg.recap_pcm:
            out += seg.recap_pcm
            out += silence(seg.recap_pre_pause)
        for r in range(plan.repeat):
            out += seg.pcm
            if r < plan.repeat - 1:
                out += silence(plan.repeat_gap)
        out += silence(seg.pause_sec)
    out += silence(plan.tail)
    return bytes(out)


def fmt_time(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def wpm_tag(wpm: float) -> str:
    """Ярлык темпа для имени файла: 20 -> '20-wpm', 17.5 -> '17.5-wpm'."""
    value = f"{wpm:g}"
    return f"{value}-wpm"


def duration_tag(seconds: float) -> str:
    """Ярлык длительности для имени файла (без ':' — недопустим в Windows)."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def default_output_path(input_path: str, *, wpm: float, duration_sec: float | None) -> Path:
    """Имя выходного файла по умолчанию: рядом с текстом, с припиской темпа
    или длительности — например, Exercise-100.txt -> Exercise-100-20-wpm.wav."""
    src = Path(input_path)
    tag = duration_tag(duration_sec) if duration_sec is not None else wpm_tag(wpm)
    return src.with_name(f"{src.stem}-{tag}.wav")


def parse_duration(spec: str) -> float:
    """Разбирает желаемую длительность в секунды.

    Форматы: "4" / "4m" / "4min" — минуты, "240s" — секунды,
    "MM:SS" или "H:MM:SS".
    """
    s = spec.strip().lower()
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
            raise ValueError(f"Не понимаю длительность «{spec}». "
                             f"Форматы: 4, 4m, 240s, 4:30, 1:02:30.")
        nums = [int(p) for p in parts]
        h, m, sec = (0, *nums) if len(nums) == 2 else nums
        return h * 3600 + m * 60 + sec
    m = re.fullmatch(r"([\d.]+)\s*(s|sec|с|секунд[а-я]*)", s)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"([\d.]+)\s*(m|min|мин|минут[а-я]*)?", s)
    if m:
        return float(m.group(1)) * 60
    raise ValueError(f"Не понимаю длительность «{spec}». "
                     f"Форматы: 4, 4m, 240s, 4:30, 1:02:30.")


def print_tts_script(plan: Plan) -> None:
    """Построчно печатает точную последовательность вызовов TTS и пауз —
    именно то, что реально попадёт в аудио, в порядке воспроизведения."""
    n = 0
    print(f"     [тишина {plan.lead_in:.1f} с]")
    for seg in plan.segments:
        if seg.recap:
            n += 1
            print(f"{n:>3}. TTS (рекап)  «{seg.sentence_text}»")
            print(f"     [тишина {seg.recap_pre_pause:.1f} с]")
        for r in range(plan.repeat):
            n += 1
            suffix = f"  (повтор {r + 1}/{plan.repeat})" if plan.repeat > 1 else ""
            print(f"{n:>3}. TTS{suffix}  «{seg.text}»")
            if r < plan.repeat - 1:
                print(f"     [тишина {plan.repeat_gap:.1f} с]")
        print(f"     [тишина {seg.pause_sec:.1f} с]")
    print(f"     [тишина {plan.tail:.1f} с]")


# ===========================================================================
# CLI
# ===========================================================================


API_KEY_FILE = Path(__file__).resolve().parent / "api-key.txt"


def read_api_key_file() -> str | None:
    try:
        return API_KEY_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def build_backend(args) -> Backend:
    if args.backend == "yandex":
        key = args.api_key or os.environ.get("YANDEX_API_KEY") or read_api_key_file()
        if not key:
            raise SynthError(
                "Не задан ключ Yandex SpeechKit. Передайте --api-key, "
                "установите переменную окружения YANDEX_API_KEY, либо "
                f"положите ключ в файл {API_KEY_FILE.name} рядом со скриптом "
                "(либо используйте --backend sapi для офлайн-синтеза)."
            )
        return YandexBackend(
            api_key=key, voice=args.voice or "alena", role=args.role,
            emotion=args.emotion, folder_id=args.folder_id,
        )
    return SapiBackend(voice=args.voice)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Генерирует аудио-диктант из текста с заданным темпом (слов в минуту).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="?", help="файл с текстом (UTF-8); '-' — читать stdin")
    p.add_argument("-o", "--output",
                   help="выходной файл .wav (или .mp3, если есть ffmpeg); "
                        "если не задан, берётся путь входного файла с припиской "
                        "темпа/длительности, напр. Exercise-100-20-wpm.wav")
    p.add_argument("--wpm", type=float, default=None,
                   help="целевой темп диктанта, слов в минуту (по умолчанию 55, "
                        "если не задан --duration)")
    p.add_argument("--duration",
                   help="вместо --wpm: уложить весь диктант в заданную "
                        "длительность (4, 4m, 240s, 4:30, 1:02:30) — темп "
                        "посчитается из числа слов автоматически")
    p.add_argument("--speed", type=float, default=1.0,
                   help="скорость самой речи диктора, 1.0 — норма (0.5–2.0)")

    p.add_argument("--backend", choices=["yandex", "sapi"], default="yandex",
                   help="движок синтеза (по умолчанию yandex)")
    p.add_argument("--api-key",
                   help="Api-Key Yandex Cloud (иначе берётся YANDEX_API_KEY, "
                        "иначе — содержимое api-key.txt рядом со скриптом)")
    p.add_argument("--folder-id", help="folderId Yandex Cloud (обычно не нужен для Api-Key)")
    p.add_argument("--voice", help="голос: yandex — alena/filipp/…; sapi — имя голоса Windows")
    p.add_argument("--role", help="амплуа голоса Yandex (neutral, good, friendly…)")
    p.add_argument("--emotion", help="устаревший параметр эмоции Yandex")
    p.add_argument("--list-voices", action="store_true",
                   help="показать голоса и выйти")

    p.add_argument("--max-words", type=int, default=5,
                   help="максимум слов в отрезке для чтения (по умолчанию 9)")
    p.add_argument("--min-words", type=int, default=2,
                   help="минимум слов в отрезке (по умолчанию 2)")
    p.add_argument("--repeat", type=int, default=1,
                   help="сколько раз читать каждый отрезок (по умолчанию 1)")
    p.add_argument("--repeat-gap", type=float, default=0.8,
                   help="пауза между повторами отрезка, с (по умолчанию 0.8)")
    p.add_argument("--min-pause", type=float, default=1.0,
                   help="минимальная пауза после отрезка, с (по умолчанию 1.0)")
    p.add_argument("--sentence-extra", type=float, default=1.0,
                   help="добавка к паузе в конце предложения, с (по умолчанию 1.0)")
    p.add_argument("--clause-extra", type=float, default=0.4,
                   help="добавка к паузе после запятой/тире/двоеточия внутри "
                        "предложения, с (по умолчанию 0.4)")

    p.add_argument("--recap", dest="recap", action="store_true", default=True,
                   help="перед разбитым на части предложением сначала "
                        "прочитать фразу целиком (включено по умолчанию)")
    p.add_argument("--no-recap", dest="recap", action="store_false",
                   help="не читать фразу целиком перед надиктовкой по частям")
    p.add_argument("--recap-pre-pause", type=float, default=1.5,
                   help="пауза после рекапа, перед началом надиктовки по "
                        "частям, с (по умолчанию 1.5)")

    p.add_argument("--lead-in", type=float, default=1.0,
                   help="тишина в начале файла, с")
    p.add_argument("--tail", type=float, default=3.0,
                   help="тишина в конце файла, с")
    p.add_argument("--full-read", choices=["none", "before", "after", "both"],
                   default="none",
                   help="прочитать текст целиком до/после диктовки (не влияет на WPM диктовки)")
    p.add_argument("--full-read-pause", type=float, default=4.0,
                   help="пауза вокруг сплошного чтения, с")

    p.add_argument("--no-trim", action="store_true",
                   help="не срезать тишину по краям синтезированных отрезков")
    p.add_argument("--cache-dir", default=".tts_cache",
                   help="каталог кэша синтеза ('' — выключить)")
    p.add_argument("--dry-run", action="store_true",
                   help="показать разбивку и тайминг без синтеза")
    p.add_argument("--script", action="store_true",
                   help="вывести подробный TTS-скрипт: точный текст и паузы "
                        "по порядку воспроизведения")

    p.add_argument("--mp3-bitrate", default="128k", help="битрейт для .mp3")

    args = p.parse_args(argv)

    if args.list_voices:
        if args.backend == "sapi":
            for name, culture, gender in SapiBackend.list_voices():
                print(f"{name}\t{culture}\t{gender}")
        else:
            print("Голоса Yandex SpeechKit (ru-RU):")
            print("  женские: alena, jane, oksana, omazh, dasha, julia, lera, masha, marina")
            print("  мужские: filipp, ermil, zahar, madirus, alexander, kirill, anton")
            print("  Для alena/filipp доступен --role (neutral, good, strict, friendly…).")
        return 0

    if not args.input:
        p.error("не указан входной файл с текстом")
    if not args.dry_run and not args.output and args.input == "-":
        p.error("для текста из stdin имя выходного файла не вывести автоматически — "
                "укажите -o явно")
    if args.wpm is not None and args.duration is not None:
        p.error("нельзя одновременно указывать --wpm и --duration")
    duration_sec = None
    if args.duration is not None:
        try:
            duration_sec = parse_duration(args.duration)
        except ValueError as exc:
            p.error(str(exc))

    text = (sys.stdin.read() if args.input == "-"
            else Path(args.input).read_text(encoding="utf-8"))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        print("Входной текст пуст.", file=sys.stderr)
        return 1

    segments = build_segments(text, args.max_words, args.min_words)
    if not segments:
        print("Не удалось выделить ни одного отрезка.", file=sys.stderr)
        return 1

    if duration_sec is not None:
        words_total = sum(s.words for s in segments)
        wpm = words_total / (duration_sec / 60)
    else:
        wpm = args.wpm if args.wpm is not None else 55.0

    if not args.output and args.input != "-":
        args.output = str(default_output_path(
            args.input, wpm=wpm, duration_sec=duration_sec,
        ))

    # Предложения из 2+ отрезков получают рекап — фразу целиком перед
    # надиктовкой по частям. Она озвучивается ОДНИМ вызовом TTS по полному
    # тексту предложения, а не склейкой уже нарезанных отрезков — иначе на
    # месте будущих запятых/тире были бы слышны лишние механические паузы.
    recap_groups = [g for g in sentence_groups(segments) if len(g) >= 2] if args.recap else []
    for group in recap_groups:
        group[0].recap = True
        group[0].recap_pre_pause = args.recap_pre_pause

    # --- длительность речи: точно (синтез) или оценкой (dry-run) -----------
    if args.dry_run:
        for seg in segments:
            seg.speech_sec = estimate_speech_sec(seg.text, args.speed)
        for group in recap_groups:
            group[0].recap_speech_sec = estimate_speech_sec(group[0].sentence_text, args.speed)
    else:
        backend = build_backend(args)
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        cache = SynthCache(backend, cache_dir)
        total_calls = len(segments) + len(recap_groups)
        done = 0
        for seg in segments:
            done += 1
            print(f"\rСинтез {done}/{total_calls}…", end="", file=sys.stderr, flush=True)
            pcm = cache.get(seg.text, args.speed)
            if not args.no_trim:
                pcm = trim_silence(pcm)
            seg.pcm = pcm
            seg.speech_sec = duration_of(pcm)
        for group in recap_groups:
            done += 1
            print(f"\rСинтез {done}/{total_calls}…", end="", file=sys.stderr, flush=True)
            first = group[0]
            try:
                pcm = cache.get(first.sentence_text, args.speed)
                if not args.no_trim:
                    pcm = trim_silence(pcm)
            except SynthError:
                # Запасной вариант (например, предложение длиннее лимита
                # API) — склеиваем уже готовые отрезки без лишних пауз.
                pcm = b"".join(s.pcm for s in group)
            first.recap_pcm = pcm
            first.recap_speech_sec = duration_of(pcm)
        print(f"\rСинтез завершён: {total_calls} вызовов "
              f"(из кэша {cache.hits}, новых {cache.misses}).", file=sys.stderr)

    plan, warning = plan_pauses(
        segments, wpm, repeat=args.repeat, repeat_gap=args.repeat_gap,
        min_pause=args.min_pause, sentence_extra=args.sentence_extra,
        clause_extra=args.clause_extra, lead_in=args.lead_in, tail=args.tail,
    )

    if args.script:
        print("=== TTS-скрипт ===")
        print_tts_script(plan)
        print()

    # --- отчёт -------------------------------------------------------------
    width = max(len(str(len(segments))), 2)
    recap_count = 0
    for i, seg in enumerate(plan.segments, 1):
        if seg.recap:
            recap_count += 1
            print(f"{'':>{width}}   ↺ фраза целиком ({seg.recap_speech_sec:.1f} с)"
                  f" + пауза {seg.recap_pre_pause:.1f} с")
        mark = "¶" if seg.is_sentence_end else "·"
        print(f"{i:>{width}} {mark} [{seg.words:>2} сл, речь {seg.speech_sec:4.1f} с, "
              f"пауза {seg.pause_sec:4.1f} с] {seg.text}")

    print()
    print(f"Слов:            {plan.words}")
    print(f"Отрезков:        {len(plan.segments)}"
          f" (предложений: {plan.segments[-1].sentence_index + 1})")
    print(f"Речь:            {fmt_time(plan.speech_total)}")
    print(f"Паузы:           {fmt_time(plan.pause_total)}")
    if recap_count:
        print(f"Рекапов фразы:   {recap_count} "
              f"({fmt_time(plan.recap_pre_total + plan.recap_speech_total)})")
    print(f"Длительность:    {fmt_time(plan.total)}")
    if duration_sec is not None:
        print(f"Темп:            {plan.actual_wpm:.1f} сл/мин "
              f"(чтобы уложиться в {fmt_time(duration_sec)} → {wpm:.1f} сл/мин)")
    else:
        print(f"Темп:            {plan.actual_wpm:.1f} сл/мин (цель {wpm:g})")
    if warning:
        print(f"\nВНИМАНИЕ: {warning}", file=sys.stderr)

    if args.dry_run:
        print("\n(--dry-run: длительность речи оценена приблизительно, "
              "синтез не выполнялся)")
        return 0

    # --- сборка ------------------------------------------------------------
    pcm = render(plan)

    if args.full_read != "none":
        full = cache.get(text, args.speed) if len(text) <= 4500 else None
        if full is None:
            # Длинный текст — склеиваем сплошное чтение из уже готовых отрезков.
            gap = silence(0.35)
            full = gap.join(seg.pcm for seg in plan.segments)
        elif not args.no_trim:
            full = trim_silence(full)
        pad = silence(args.full_read_pause)
        if args.full_read in ("before", "both"):
            pcm = silence(args.lead_in) + full + pad + pcm
        if args.full_read in ("after", "both"):
            pcm = pcm + full + silence(args.tail)
        print(f"Со сплошным чтением: {fmt_time(duration_of(pcm))}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".mp3":
        tmp_wav = out.with_suffix(".tmp.wav")
        write_wav(tmp_wav, pcm)
        try:
            encode_mp3(tmp_wav, out, args.mp3_bitrate)
        finally:
            tmp_wav.unlink(missing_ok=True)
    else:
        write_wav(out, pcm)

    print(f"\nГотово: {out}  ({fmt_time(duration_of(pcm))})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SynthError as exc:
        print(f"\nОшибка: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nПрервано.", file=sys.stderr)
        sys.exit(130)
