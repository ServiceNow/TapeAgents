import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Optional

import ffmpeg
import webvtt
import whisper
import yt_dlp
from whisper.utils import get_writer

from tapeagents.config import ATTACHMENT_DEFAULT_DIR
from tapeagents.core import Action, Observation
from tapeagents.steps import VideoObservation, WatchVideoAction
from tapeagents.tools.base import Tool

logger = logging.getLogger(__name__)


def get_video_observation(
    url: str,
    output_dir: str,
    start_time: str = "",
    end_time: str = "",
) -> VideoObservation:
    try:
        video_path, thumbnail_path = download_video(url, output_dir)
        video_path_trimmed = trim_video(video_path, start_time, end_time)
        video_contact_sheet_paths = generate_contact_sheets_from_video(
            video_path, video_path_trimmed=video_path_trimmed, start_time=start_time, end_time=end_time
        )
        try:
            subtitle_path = transcribe_audio(video_path, video_path_trimmed, start_time=start_time, end_time=end_time)
        except Exception as e:
            logger.exception(f"Error while transcribing audio: {e}")
            subtitle_path = None
        try:
            subtitle_text = extract_text_from_vtt(subtitle_path, start_time, end_time)
        except Exception as e:
            logger.exception(f"Error while extracting text from VTT: {e}")
            subtitle_text = None
        error = None
    except Exception as e:
        logger.exception(f"Error while watching video: {e}")
        raise e
    video_observation = VideoObservation(
        attachment_dir=output_dir,
        video_path=Path(video_path_trimmed).relative_to(output_dir).as_posix(),
        video_contact_sheet_paths=[Path(path).relative_to(output_dir).as_posix() for path in video_contact_sheet_paths],
        thumbnail_path=Path(thumbnail_path).relative_to(output_dir).as_posix(),
        subtitle_path=Path(subtitle_path).relative_to(output_dir).as_posix(),
        subtitle_text=subtitle_text,
        error=error,
    )
    return video_observation


def download_video(url: str, output_dir: str) -> str:
    if "youtube" in url:
        video = download_video_youtube(url, output_dir)
        time.sleep(2)
        return video
    else:
        raise NotImplementedError("Only youtube videos are supported at the moment")


def trim_video(video_path: str, start_time: str, end_time: str) -> str:
    if not video_path:
        raise ValueError("video_path is required")

    if not start_time and not end_time:
        return video_path

    if not start_time:
        start_time = "00:00:00"

    if not end_time:
        probe = ffmpeg.probe(video_path)
        video_duration = float(probe["streams"][0]["duration"])
        end_time = str(video_duration)

    if start_time == end_time:
        start_time = str(float(end_time) - 0.5)
        end_time = str(float(end_time) + 0.5)

    try:
        logger.info("Trimming video")
        video_base_path, ext = os.path.splitext(video_path)
        trimmed_video_path = f"{video_base_path}_{start_time.replace(':', '-')}_{end_time.replace(':', '-')}{ext}"
        ffmpeg.input(video_path).output(
            trimmed_video_path,
            ss=start_time,
            to=end_time,
        ).run(overwrite_output=True)
    except ffmpeg.Error as e:
        raise e
    return trimmed_video_path


def download_video_youtube(url: str, output_dir: str) -> str:
    ydl_opts = {
        "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",  # Download best mp4 format available or any other best if no mp4 available
        "formatsort": "res:360",  # Video available with the largest resolution but no better than 360p
        "writethumbnail": True,
        "writesubtitles": True,  # Official docs mention write-subs
        "writeautomaticsub": True,  # Official docs mention write-auto-subs
        "paths": {"home": output_dir},
        "outtmpl": "%(id)s.%(ext)s",  # Output file name template
        "logger": YTDLogger(),
        "progress_hooks": [ytd_progress_hook],
    }

    video_id = get_video_id(url)
    video_path = find_file(output_dir, video_id, (".mp4", ".webm"))
    thumbnail_path = find_file(output_dir, video_id, (".webp", ".jpg", ".jpeg", ".png"), strict=True)
    if not video_path:  # video not in cache
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download(url)
            except yt_dlp.utils.YoutubeDLError as e:
                # Output list format for debug
                ydl_opts["listformats"] = True
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download(url)
                raise e
        video_path = find_file(output_dir, video_id, (".mp4", ".webm"))
        thumbnail_path = find_file(output_dir, video_id, (".webp", ".jpg", ".jpeg", ".png"), strict=True)
    return video_path, thumbnail_path


def get_video_id(url: str) -> str:
    """
    Extract the video ID from a YouTube URL.
    Supports various YouTube URL formats.
    """
    video_id = None
    patterns = [
        r"v=([^&]+)",  # e.g. https://www.youtube.com/watch?v=VIDEO_ID
        r"youtu\.be/([^?&]+)",  # e.g. https://youtu.be/VIDEO_ID
        r"youtube\.com/embed/([^?&]+)",  # e.g. https://www.youtube.com/embed/VIDEO_ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    return video_id


def find_file(dir: str, filename: str, file_extentions: tuple, strict: bool = False) -> Optional[str]:
    if strict:
        for file_extention in file_extentions:
            file = f"{filename}{file_extention}"
            file_path = os.path.join(dir, file)
            if os.path.exists(file_path):
                return file_path
        return None

    for file in os.listdir(dir):
        if file.startswith(filename) and file.endswith(file_extentions):
            return os.path.join(dir, file)
    return None


def generate_contact_sheets_from_video(
    video_path: str,
    video_path_trimmed: str,
    max_contact_sheets: int = 1,
    frame_interval_seconds: int = 0,  # 0 means auto frame interval
    start_time: str = "",
    end_time: str = "",
) -> list[str]:
    # Optimized grid for 512px x 512px: https://platform.openai.com/docs/guides/vision
    nb_tile_x = 3
    nb_tile_y = 5
    scale = "320:180"
    format = "png"
    output_suffix = "_contact-sheet_"

    # Return existing contact sheets if they exist
    output_paths = []
    output_dir = os.path.dirname(video_path_trimmed)
    file_base_path, _ = os.path.splitext(os.path.basename(video_path_trimmed))
    for file in os.listdir(output_dir):
        if re.search(r"{}{}\d+".format(file_base_path, output_suffix), file):
            output_paths.append(os.path.join(output_dir, file))
    if len(output_paths) > 0:
        logger.info(f"Use cached contact sheets: {output_paths}")
        return output_paths

    # Determine the number of frames and contact sheets needed
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        logger.error(f"ffprobe error: {e.stderr.decode('utf-8')}")
        raise e
    video_duration = float(probe["streams"][0]["duration"])  # Video duration in seconds
    fps = eval(probe["streams"][0]["r_frame_rate"])  # Extract FPS

    start_time_seconds = time_to_seconds(start_time) if start_time else 0
    end_time_seconds = time_to_seconds(end_time) if end_time else video_duration
    end_time_seconds = min(end_time_seconds, video_duration)

    duration = end_time_seconds - start_time_seconds
    if frame_interval_seconds == 0:  # Auto frame interval
        frame_interval_seconds = duration / (max_contact_sheets * nb_tile_x * nb_tile_y)
    elif duration <= frame_interval_seconds:  # If the video is shorter than the frame interval, use the whole video
        frame_interval_seconds = duration
    logger.info(f"Frame interval (s): {frame_interval_seconds}")
    total_nb_frames = int(duration / frame_interval_seconds)
    nb_tile_y = min(nb_tile_y, math.ceil(total_nb_frames / nb_tile_x))
    nb_tile_x = min(nb_tile_x, math.ceil(total_nb_frames / nb_tile_y))
    frames_per_contact_sheet = nb_tile_x * nb_tile_y
    total_contact_sheets = math.ceil(total_nb_frames / frames_per_contact_sheet)
    frames_per_interval = int(fps * frame_interval_seconds)

    # Generate contact sheets
    vf = """drawtext=text='%{pts\\:hms}'
            :x='(main_w-text_w)/2'
            :y='(main_h-text_h)'
            :fontcolor='Yellow'
            :fontsize='(main_h/12)'
            :boxcolor='Black':box=1
            """  # Add a timestamp to each frame
    vf += f",thumbnail=n={frames_per_interval}"  # Pick one of the most representative frames in sequences of nb_frames consecutive frames
    vf += f",scale={scale}"  # Resize the frames to a specific size
    vf += ",pad=iw+4:ih+4:2:2:color=black"  # Add borders around each frame
    vf += f",tile={nb_tile_x}x{nb_tile_y}"  # Arrange the frames in a grid

    logger.info("Generating contact sheet from video")
    output_paths = []
    video_base_path, _ = os.path.splitext(video_path_trimmed)
    for i in range(total_contact_sheets):
        start_frame = i * frame_interval_seconds * frames_per_contact_sheet
        ss = start_time_seconds + start_frame
        to = ss + duration
        output_path = f"{video_base_path}{output_suffix}{i+1}.{format}"
        try:
            cmd = ffmpeg.input(video_path, to=to).output(
                output_path,
                ss=ss,
                vframes=1,
                vf=vf,
            )
            logger.info(f"FFMPEG ARGS: {cmd.get_args()}")
            cmd.run(overwrite_output=True)
            output_paths.append(output_path)
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")
            raise e
    logger.info(f"Generated contact sheets: {output_paths}")
    return output_paths


def extract_audio(
    video_path: str,
    audio_path: Optional[str] = None,
    audio_bitrate: str = "128k",
    acodec: str = "mp3",
    max_duration: int = 300,  # 5 minutes
) -> str:
    if not audio_path:
        video_base_path, _ = os.path.splitext(video_path)
        audio_path = f"{video_base_path}.{acodec}"

    if os.path.exists(audio_path):
        return audio_path

    probe = ffmpeg.probe(video_path)
    video_duration = float(probe["streams"][0]["duration"])

    end_time_seconds = min(video_duration, max_duration)

    try:
        logger.info("Extracting audio from video")
        ffmpeg_input = ffmpeg.input(video_path, to=end_time_seconds)
        ffmpeg_input.output(
            audio_path,
            audio_bitrate=audio_bitrate,
            acodec=acodec,
        ).run(overwrite_output=True)
    except ffmpeg.Error as e:
        raise e
    return audio_path


def transcribe_audio(video_path: str, video_path_trimmed: str, start_time: str = "", end_time: str = "") -> str:
    if start_time:
        # Check if a VTT file already exists for the trimmed video
        dir_trimmed = os.path.dirname(video_path_trimmed)
        video_id_trimmed, _ = os.path.splitext(os.path.basename(video_path_trimmed))
        vtt_file = find_file(dir_trimmed, video_id_trimmed, (".vtt",), strict=True)
        if vtt_file:
            ensure_vtt_format(vtt_file)
            return vtt_file

        audio_path = find_file(dir_trimmed, video_id_trimmed, (".mp3",), strict=True)
    else:
        # Check if a VTT file already exists
        dir = os.path.dirname(video_path)
        video_id, _ = os.path.splitext(os.path.basename(video_path))
        vtt_file = find_file(dir, video_id, (".vtt",), strict=True)
        if vtt_file:
            ensure_vtt_format(vtt_file)
            return vtt_file

        audio_path = find_file(dir, video_id, (".mp3",), strict=True)

    if audio_path is None:
        audio_path = extract_audio(video_path_trimmed)

    vtt_path = audio_path.replace(".mp3", ".vtt")
    model = whisper.load_model("turbo")
    result = model.transcribe(audio_path)
    output_directory = os.path.dirname(audio_path)

    vtt_writer = get_writer("vtt", output_directory)
    vtt_writer(result, audio_path)
    vtt_path = increase_vtt_timestamps(vtt_path, offset_time=start_time)
    return vtt_path


def increase_vtt_timestamps(vtt_path: str, offset_time: str) -> str:
    offset_seconds = time_to_seconds(offset_time) if offset_time else 0
    if offset_seconds > 0:
        vtt = webvtt.read(vtt_path)
        for caption in vtt:
            caption.start = seconds_to_time(time_to_seconds(caption.start) + offset_seconds)
            caption.end = seconds_to_time(time_to_seconds(caption.end) + offset_seconds)
        vtt.save(vtt_path)
    return vtt_path


def extract_text_from_vtt(vtt_path: str, start_time: str = "", end_time: str = "") -> str:
    start_time = ensure_milliseconds(start_time)
    end_time = ensure_milliseconds(end_time)
    text = []

    vtt = webvtt.read(vtt_path)

    for caption in vtt.iter_slice(start=start_time, end=end_time):
        text.append(f"{str(caption.start)}: {str(caption.text)}")
    return "\n".join(text)


def ensure_vtt_format(vtt_path: str) -> None:
    with open(vtt_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(vtt_path, "w", encoding="utf-8") as file:
        for line in lines:
            if line.strip() == "" and line.startswith(" "):  # Do not write line that contains only space
                continue
            else:
                file.write(line)


def ensure_milliseconds(time: str) -> None:
    """Ensure that the time string has milliseconds HH:MM:SS.mmm"""
    if not time:
        return time
    # validate time format
    if not re.match(r"^\d{2}:\d{2}:\d{2}(\.\d{3})?$", time):
        raise ValueError("Invalid time format:", time)
    # add .mmm if not included
    if time[-4] != ".":
        time += ".000"
    return time


def seconds_to_time(seconds: float) -> str:
    """Convert seconds SS to time string HH:MM:SS.mmm"""
    if type(seconds) is not float:
        try:
            seconds = float(seconds)
        except ValueError:
            raise ValueError("Invalid seconds format")
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"


def time_to_seconds(time_str: str) -> float:
    """Convert a time string HH:MM:SS(.mmm) and MM:SS().mmm) to seconds SS"""
    parts = list(map(float, re.split("[:.]", time_str)))
    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return int(parts[0]) * 60 + parts[1]
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + parts[2]
    elif len(parts) == 4:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + parts[2] + parts[3] / 1000
    else:
        raise ValueError("Invalid time format")


class YTDLogger(object):
    def debug(self, msg: str) -> None:
        # For compatibility with youtube-dl, both debug and info are passed into debug
        # You can distinguish them by the prefix '[debug] '
        if msg.startswith("[debug] "):
            logger.debug(msg)
        else:
            self.info(msg)

    def info(self, msg: str) -> None:
        logger.info(msg)
        pass

    def warning(self, msg: str) -> None:
        logger.warning(msg)
        pass

    def error(self, msg: str) -> None:
        logger.error(msg)


def ytd_progress_hook(d: dict) -> None:
    if d["status"] == "finished":
        logger.info("Done downloading, now converting ...")


class VideoReader(Tool):
    """
    Opens video from a youtube URL.
    Can access the video content, thumbnail, subtitles and audio.
    """

    action: type[Action] = WatchVideoAction
    observation: type[Observation] = VideoObservation
    cached: bool = True
    exp_path: str

    def execute_action(self, action: WatchVideoAction) -> VideoObservation:
        attachment_dir = os.path.join(self.exp_path, ATTACHMENT_DEFAULT_DIR)
        os.makedirs(attachment_dir, exist_ok=True)
        return get_video_observation(action.video_url, attachment_dir, action.start_time, action.end_time)
