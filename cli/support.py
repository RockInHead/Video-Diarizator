import os
import logging
from typing import Optional, Tuple


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_file_paths() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Запрашивает у пользователя тип объекта, наличие видеофайла и текстового описания.
    Для видео файлов проверяется, что расширение .mp4 или .avi.
    Для текстового описания проверяется, что расширение .txt.
    Если введено "0", функция возвращает (None, None) для возврата в главное меню.

    :return: Кортеж (type_obj, video_path, txt_path), где video_path и txt_path – корректные пути к файлам или None.
    """
    video_path = None
    txt_path = None
    type_obj = None


    # Запрос наличия типа добавляемого объекта
    choise_type = input("Тип добавляемого объекта? (1 - кандидат, 2 - проект): ").strip()
    while True:
        if choise_type == "1":
            type_obj == "kandidate"
            break
        
        elif choise_type == "2":
            type_obj == "project"
            break

        elif choise_type == "0":
            return (None, None, None)

        else:
            print("Неверный ввод для видео. Возврат в главное меню.")


    # Запрос наличия видеофайла
    choice_video = input("Есть ли видео файл? (1 - да, 2 - нет): ").strip()
    if choice_video == "1":
        while True:
            path = input("Введите полный путь к видео файлу (mp4 или avi) или 0 для возврата: ").strip()
            if path == "0":
                return (None, None, None)
            
            if not os.path.exists(path):
                print("Указанный файл не существует. Попробуйте снова.")
                continue
            
            ext = os.path.splitext(path)[1].lower()
            if ext not in ['.mp4', '.avi']:
                print("Неверное расширение. Допустимы только .mp4 и .avi. Попробуйте снова.")
                continue
            
            video_path = path
            print(f"Видео файл выбран: {video_path}")
            break
    
    elif choice_video == "2":
        print("Видео файл не используется.")
    
    else:
        print("Неверный ввод для видео. Возврат в главное меню.")
        return (None, None, None)

    # Запрос наличия текстового описания
    choice_txt = input("Есть ли текстовое описание? (1 - да, 2 - нет): ").strip()
    if choice_txt == "1":
        while True:
            path = input("Введите полный путь к текстовому файлу (.txt) или 0 для возврата: ").strip()
            if path == "0":
                return (None, None, None)
            
            if not os.path.exists(path):
                print("Указанный файл не существует. Попробуйте снова.")
                continue
            
            ext = os.path.splitext(path)[1].lower()
            if ext != '.txt':
                print("Неверное расширение. Допустим только .txt. Попробуйте снова.")
                continue
            
            txt_path = path
            print(f"Текстовый файл выбран: {txt_path}")
            break
    
    elif choice_txt == "2":
        print("Текстовое описание не используется.")
    
    else:
        print("Неверный ввод для текстового файла. Возврат в главное меню.")
        return (None, None, None)

    return (type_obj, video_path, txt_path)
