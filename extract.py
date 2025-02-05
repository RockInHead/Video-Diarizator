import ffmpeg
import os

def start_extract_audio(video_path, audio_dir_path):
    """Извлекает аудио из видео."""
    
    video_file_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_file_name = f"{video_file_name}.mp3"  
    audio_path = os.path.join(audio_dir_path, audio_file_name)
    
    try:
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
    except ffmpeg.Error as e:
        print(f"Ошибка при извлечении аудио: {e}")
        return None
    return audio_path
