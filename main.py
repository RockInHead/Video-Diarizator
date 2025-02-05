from extract import start_extract_audio
from diarizator.diarize import start_diarize
from assistant import start_assistant

def main(video_path, audio_dir_path, text_dir_path, ai_text_dir_path):
    
    # Извлекаем аудио из видео.
    audio_path = start_extract_audio(video_path, audio_dir_path)
    
    # Выполняем диаризацию. Сохраняем путь до файл с диаризацией диалога.
    diarized_text_path = start_diarize(audio = audio_path, model_name = "large", language= "ru", text_dir_path = text_dir_path)
    
    #Отправляем файл с диаризацией в ChatGPT.
    start_assistant(diarized_text_path, ai_text_dir_path)
    
if __name__ == "__main__":
    video_path = r"D:\AI Strategy\Транскрибатор\assets\Собеседование 3min.mp4"
    audio_dir_path = r"D:\AI Strategy\Транскрибатор\result"
    diarized_text_dir_path = r"D:\AI Strategy\Транскрибатор\result"
    ai_text_dir_path = r"D:\AI Strategy\Транскрибатор\result"

    main(video_path, audio_dir_path, diarized_text_dir_path, ai_text_dir_path)