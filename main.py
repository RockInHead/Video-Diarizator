import ffmpeg
from diarizator.diarize import start_diarize
from assistant import start_assistant
import os

def extract_audio(video_path, audio_dir_path):
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

def main(video_path, audio_dir_path, text_dir_path, ai_text_dir_path):
    
    # Извлекаем аудио из видео
    audio_path = extract_audio(video_path, audio_dir_path)
    
    # Выполняем диаризацию. Сохраняем путь до файл с диаризацией диалога.
    diarized_text_path = start_diarize(audio = audio_path, model_name = "large", language= "ru", text_dir_path = text_dir_path)
    
    #Отправляем файл с диаризацией в ChatGPT.
    start_assistant(diarized_text_path, ai_text_dir_path)
    

if __name__ == "__main__":
    video_path = r"D:\AI Strategy\Транскрибатор\assets\Собеседование 7min.mp4"
    audio_dir_path = r"D:\AI Strategy\Транскрибатор\result"
    diarized_text_dir_path = r"D:\AI Strategy\Транскрибатор\result"
    ai_text_dir_path = r"D:\AI Strategy\Транскрибатор\result"

    main(video_path, audio_dir_path, diarized_text_dir_path, ai_text_dir_path)