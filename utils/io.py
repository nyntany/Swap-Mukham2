import os
import cv2
import glob
import shutil
import subprocess
from datetime import datetime


image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "ico", "webp"]

def get_images_from_directory(directory_path):
    file_paths =[]
    for file_path in glob.glob(os.path.join(directory_path, "*")):
        if any(file_path.lower().endswith(ext) for ext in image_extensions):
            file_paths.append(file_path)
    file_paths.sort()
    return file_paths


def open_directory(path=None):
    if path is None:
        return
    try:
        os.startfile(path)
    except:
        subprocess.Popen(["xdg-open", path])


def copy_files_to_directory(files, destination):
    file_paths = []
    for file_path in files:
        new_file_path = shutil.copy(file_path, destination)
        file_paths.append(new_file_path)
    return file_paths


def create_directory(directory_path, remove_existing=True):
    if os.path.exists(directory_path) and remove_existing:
        shutil.rmtree(directory_path)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
        return directory_path
    else:
        counter = 1
        while True:
            new_directory_path = f"{directory_path}_{counter}"
            if not os.path.exists(new_directory_path):
                os.mkdir(new_directory_path)
                return new_directory_path
            counter += 1


def add_datetime_to_filename(filename):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    file_name, file_extension = os.path.splitext(filename)
    new_filename = f"{file_name}_{formatted_datetime}{file_extension}"
    return new_filename


def get_single_video_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = min(int(frame_index), total_frames-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    valid_frame, frame = cap.read()
    cap.release()
    if valid_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def ffmpeg_extract_frames(video_path, destination, remove_existing=True, fps=30, name='frame_%d.jpg', ffmpeg_path=None):
    ffmpeg_path = 'ffmpeg' if ffmpeg_path is None else ffmpeg_path
    destination = create_directory(destination, remove_existing=remove_existing)
    cmd = [
        ffmpeg_path,
        '-loglevel', 'info',
        '-hwaccel', 'auto',
        '-i', video_path,
        '-q:v', '3',
        '-pix_fmt', 'rgb24',
        '-vf', 'fps=' + str(fps),
        '-y',
        os.path.join(destination, name)
    ]
    process = subprocess.Popen(cmd)
    process.communicate()
    if process.returncode == 0:
        return True, get_images_from_directory(destination)
    else:
        print(f"Error: Failed to extract video.")
    return False, None


def ffmpeg_merge_frames(sequence_directory, pattern, destination, fps=30, crf=18, ffmpeg_path=None):
    ffmpeg_path = 'ffmpeg' if ffmpeg_path is None else ffmpeg_path
    cmd = [
        ffmpeg_path,
        '-loglevel', 'info',
        '-hwaccel', 'auto',
        '-r', str(fps),
        # '-pattern_type', 'glob',
        '-i', os.path.join(sequence_directory, pattern),
        '-c:v', 'h264_nvenc',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        '-y', destination
    ]
    process = subprocess.Popen(cmd)
    process.communicate()
    if process.returncode == 0:
        return True, destination
    else:
        print(f"Error: Failed to merge image sequence.")
    return False, None


def ffmpeg_replace_video_segments(main_video_path, sub_clips_info, output_path, ffmpeg_path="ffmpeg"):
    ffmpeg_path = 'ffmpeg' if ffmpeg_path is None else ffmpeg_path
    filter_complex = ""

    filter_complex += f"[0:v]split=2[v0][main_end]; "
    filter_complex += f"[1:v]split={len(sub_clips_info)}{', '.join([f'[v{index + 1}]' for index in range(len(sub_clips_info))])}; "

    overlay_exprs = "".join([f"[v{index + 1}]" for index in range(len(sub_clips_info))])
    overlay_filters = f"[main_end][{overlay_exprs}]overlay=eof_action=pass[vout]; "
    filter_complex += overlay_filters

    cmd = [
        ffmpeg_path, '-i', main_video_path,
    ]

    for sub_clip_path, _, _ in sub_clips_info:
        cmd.extend(['-i', sub_clip_path])

    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[vout]',
        output_path
    ])

    subprocess.run(cmd)


def ffmpeg_mux_audio(source, target, output, ffmpeg_path=None):
    ffmpeg_path = 'ffmpeg' if ffmpeg_path is None else ffmpeg_path
    extracted_audio_path = os.path.join(os.path.dirname(output), 'extracted_audio.aac')
    cmd1 = [
        ffmpeg_path,
        '-loglevel', 'info',
        '-i', source,
        '-vn',
        '-c:a', 'aac',
        '-y',
        extracted_audio_path
    ]
    process = subprocess.Popen(cmd1)
    process.communicate()
    if process.returncode != 0:
        print(f"Error: Failed to extract audio.")
        return False, target

    cmd2 = [
        ffmpeg_path,
        '-loglevel', 'info',
        '-hwaccel', 'auto',
        '-i', target,
        '-i', extracted_audio_path,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-y', output
    ]
    process = subprocess.Popen(cmd2)
    process.communicate()
    if process.returncode == 0:
        if os.path.exists(extracted_audio_path):
            os.remove(extracted_audio_path)
        return True, output
    else:
        print(f"Error: Failed to mux audio.")
    return False, None

