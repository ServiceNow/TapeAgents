import logging
import math
import os
import re
from typing import Optional, Tuple

import ffmpeg
import webvtt
import whisper
import yt_dlp
from whisper.utils import get_writer

logger = logging.getLogger(__name__)


def watch_video(url: str, start_time: str, output_dir: str) -> Tuple[str, str, str, str, str, Optional[str]]:
    try:
        video_path, thumbnail_path = download_video(url, output_dir)
        # TODO: save video from start_time to stop_time before more processing
        video_contact_sheet_paths = generate_contact_sheets_from_video(video_path)
        audio_path = extract_audio(video_path)
        subtitle_path = transcribe_audio(audio_path)
        subtitle_text = extract_text_from_vtt(subtitle_path)
        error = None
    except Exception as e:
        logger.error(f"Error while watching video: {e}")
        raise e
    return video_path, video_contact_sheet_paths, thumbnail_path, subtitle_path, subtitle_text, error


def download_video(url: str, output_dir: str) -> str:
    if "youtube" in url:
        return download_video_youtube(url, output_dir)
    else:
        raise NotImplementedError("Only youtube videos are supported at the moment")


def download_video_youtube(url: str, output_dir: str) -> str:
    ydl_opts = {
        "format": "best",  # Select the best quality format that contains both video and audio
        "formatsort": "res:480",  # Video available with the largest resolution but no better than 480p
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
    thumbnail_path = find_file(output_dir, video_id, ("webp", ".jpg", ".jpeg", ".png"))
    if not video_path:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(url)
            video_path = find_file(output_dir, video_id, (".mp4", ".webm"))
            thumbnail_path = find_file(output_dir, video_id, ("webp", ".jpg", ".jpeg", ".png"))
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


def find_file(dir: str, filename: str, file_extentions: tuple) -> Optional[str]:
    for file in os.listdir(dir):
        if file.startswith(filename) and file.endswith(file_extentions):
            return os.path.join(dir, file)
    return None


def generate_contact_sheets_from_video(
    video_path: str,
    output_dir: Optional[str] = None,
    frame_interval_seconds: int = 5,
) -> list[str]:
    # Optimized grid for 512px x 512px: https://platform.openai.com/docs/guides/vision
    nb_tile_x = 3
    nb_tile_y = 5
    scale = "320:180"
    format = "png"
    output_suffix = "_contact-sheet_"

    if not output_dir:
        output_dir = os.path.dirname(video_path)

    # Return existing contact sheets if they exist
    output_paths = []
    for file in os.listdir(output_dir):
        base_file_name, _ = os.path.splitext(file)  # file without extension
        if file.startswith(base_file_name) and re.search(r"{}\d+".format(output_suffix), file):
            output_paths.append(os.path.join(output_dir, file))
    if len(output_paths) > 0:
        return output_paths

    # Determine the number of frames and contact sheets needed
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        logger.error(f"ffprobe error: {e.stderr.decode('utf-8')}")
        raise e
    video_duration = float(probe["streams"][0]["duration"])  # Video duration in seconds
    fps = eval(probe["streams"][0]["r_frame_rate"])  # Extract FPS
    nb_frames = fps * frame_interval_seconds
    total_frames = int(video_duration * fps)

    frames_per_contact_sheet = nb_tile_x * nb_tile_y
    total_contact_sheets = math.ceil(total_frames / (nb_frames * frames_per_contact_sheet))

    # Generate contact sheets
    vf = """drawtext=text='%{pts\\:hms}'
            :x='(main_w-text_w)/2'
            :y='(main_h-text_h)'
            :fontcolor='Yellow'
            :fontsize='(main_h/12)'
            :boxcolor='Black':box=1
            """  # Add a timestamp to each frame
    vf += f",thumbnail=n={nb_frames}"  # Pick one of the most representative frames in sequences of nb_frames consecutive frames
    vf += f",scale={scale}"  # Resize the frames to a specific size
    vf += ",pad=iw+4:ih+4:2:2:color=black"  # Add borders around each frame
    vf += f",tile={nb_tile_x}x{nb_tile_y}"  # Arrange the frames in a grid

    logger.info("Generating contact sheet from video")
    output_paths = []
    _, video_ext = os.path.splitext(video_path)
    for i in range(total_contact_sheets):
        start_frame = i * nb_frames * frames_per_contact_sheet
        output_path = video_path.replace(f".{video_ext}", f"{output_suffix}{i+1}.{format}")
        try:
            ffmpeg.input(video_path).output(
                output_path,
                ss=start_frame / fps,
                vframes=1,
                vf=vf,
            ).run(overwrite_output=True)
            output_paths.append(output_path)
        except ffmpeg.Error as e:
            logger.error(f"ffmpeg error: {e.stderr.decode('utf-8')}")
            raise e
    return output_paths


def extract_audio(
    video_path: str, audio_path: Optional[str] = None, audio_bitrate: str = "128k", acodec: str = "mp3"
) -> str:
    if not audio_path:
        _, video_ext = os.path.splitext(video_path)
        audio_path = video_path.replace(video_ext, acodec)

    if os.path.exists(audio_path):
        return audio_path

    try:
        logger.info("Extracting audio from video")
        ffmpeg.input(video_path).output(
            audio_path,
            audio_bitrate=audio_bitrate,
            acodec=acodec,
        ).run(overwrite_output=True)
    except ffmpeg.Error as e:
        raise e
    return audio_path


def transcribe_audio(audio_path: str) -> str:
    vtt_path = audio_path.replace(".mp3", ".vtt")
    if os.path.exists(vtt_path):
        return vtt_path

    model = whisper.load_model("turbo")
    result = model.transcribe(audio_path)
    output_directory = os.path.dirname(audio_path)

    vtt_writer = get_writer("vtt", output_directory)
    vtt_writer(result, audio_path)
    return vtt_path


def extract_text_from_vtt(vtt_path: str) -> str:
    vtt = webvtt.read(vtt_path)
    text = []
    for caption in vtt:
        text.append(f"{str(caption.start)}: {str(caption.text)}")
    return "\n".join(text)


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
