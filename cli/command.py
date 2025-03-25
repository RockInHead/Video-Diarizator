from cli.support import get_file_paths
from core.storage.faiss_db import FaissDB
from moduls.gpt_assist import GPTAssistant
from core.controllers.RAG_controller import RAG
import sys
import logging
import os, re
import warnings


warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_cli():
    """
    Точка входа в CLI.
    Инициализирует FAISS (загружает индексы из файлов),
    создаёт объект GPTAssistant, и запускает основной цикл команд.
    """
    try:
        FaissDB.initialize()  # Инициализируем индексы FAISS
        assistant = GPTAssistant()
        print("ChatGPT ассистент подключен успешно.")
        cli_command(assistant)
    
    except Exception as e:
        logger.error(f"Ошибка при инициализации: {e}")
        sys.exit(1)


def cli_command(assistant):
    """
    Основная функция CLI.
    Выводит меню и ожидает ввод команды пользователя.
    
    Доступные команды:
      1 - Добавить кандидата
      2 - Удалить кандидата
      3 - Показать имеющиеся данные
      4 - Подбор проектов \\ кандидатов
      10 - Загрузить тестовые данные
      0 - Выход
    """
    print_menu()
    func = {
        "1": add_object,
        "2": delete_object,
        "3": show_objects,
        "4": match_and_delete_object,
        "10": load_test_data
    }
    
    while True:
        choice = input("Введите номер команды: ").strip()
        if choice == "0":
            print("Выход из программы.")
            break
        
        select = func.get(choice)
        if select is not None:
            result = select(assistant)
            print(result)
        
        else:
            print("Команда не обнаружена.\n")
        
        print_menu()


def print_menu():
    text = """Выберите команду:
    1 - Добавить кандидата
    2 - Удалить кандидата
    3 - Показать имеющиеся данные
    4 - Подбор проектов \\ кандидатов
    10 - Загрузить тестовые данные
    0 - Выход"""
    print(text)


def add_object(assistant: GPTAssistant) -> str:
    """
    Пайплайн добавления нового объекта.
    Запрашивает у пользователя пути к видео и/или текстовому файлу через get_file_paths().
    Затем вызывает метод RAG.add_object_from_files, который выполняет полный цикл:
    извлечение текста (если задан), парсинг, нормализацию и добавление в индекс.
    Возвращает итоговое сообщение с результатом добавления.
    """
    type_obj, video_path, txt_path = get_file_paths()
    if not video_path and not txt_path:
        return "Ни один источник не выбран. Возврат в главное меню."
    
    result_message = RAG.add_object_from_files(assistant, video_path, txt_path)
    return result_message


def delete_object(assistant) -> str:
    """
    В теории - удаляет объект из базы по ID. Работоспособность не отлажена (не приоритет)
    """
    object_id = input("Введите ID объекта для удаления: ").strip()
    result = RAG.delete_object(object_id, "kandidate")
    return result


def show_objects(assistant) -> str:
    """
    Заглушка для команды показа имеющихся данных.
    Запрашивает тип (kandidate/project) и использует RAG.get_all_objects.
    """
    doc_type = input("Введите тип объектов для показа (kandidate/project): ").strip().lower()
    result = RAG.get_all_objects(doc_type)
    return result


def load_test_data(assistant) -> str:
    """
    Загружает тестовые данные из папки /app/data.
    
    Для каждого файла с расширением .txt:
      1. Считывает содержимое файла.
      2. Если файл не пустой, передаёт текст в process_text_summary (из moduls.text_processing)
         для получения структурированного словаря.
      3. Вызывает RAG.add_object для добавления объекта в индекс.
      4. Формирует итоговое сообщение для каждого файла.
      
    Возвращает итоговое сообщение.
    """
    folder = "data"
    if not os.path.exists(folder):
        return "Ошибка: папка /data не существует."
    
    files = os.listdir(folder)
    if not files:
        return "Папка /data пуста."
    
    messages = []
    for filename in files:
        if not filename.lower().endswith(".txt"):
            continue  # Обрабатываем только текстовые файлы
        
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                messages.append(f"{filename}: файл пустой.")
                continue
            
            print(f"{filename}: данные из файла получены.")
        
        except Exception as e:
            messages.append(f"{filename}: ошибка при чтении файла: {e}")
            continue
        
        try:
            from moduls.text_processing import process_text_summary
            parsed_dict = process_text_summary(content, assistant)
            
            if not parsed_dict:
                messages.append(f"{filename}: парсер вернул пустой результат.")
                continue
        
        except Exception as e:
            messages.append(f"{filename}: ошибка при обработке текста: {e}")
            continue
        
        try:
            success, result = RAG.add_object(parsed_dict)
            if not success:
                messages.append(f"{filename}: ошибка при добавлении объекта: {result}")
            
            else:
                obj_id = result
                messages.append(f"{filename}: объект успешно добавлен, ID: {obj_id}")
        
        except Exception as e:
            messages.append(f"{filename}: ошибка при добавлении объекта в базу: {e}")
    
    return "\n".join(messages)


def match_and_delete_object(assistant: GPTAssistant) -> str:
    """
    Тестовый цикл для проверки автоматизированного подбора:
      1. Запрашивает пути к файлам для добавления объекта (через get_file_paths).
      2. Добавляет объект через RAG.add_object_from_files.
      3. Извлекает ID добавленного объекта из возвращённого сообщения.
      4. Определяет тип добавленного объекта автоматически:
            - Если объект кандидат (например, type="kandidate" или "программист"), то подбираются проекты.
            - Если объект проект (например, type="project" или "проект"), то подбираются кандидаты.
      5. Выполняет подбор с использованием RAG.match_object.
      6. Удаляет добавленный объект через RAG.delete_object.
      7. Возвращает итоговое сообщение с результатами подбора и удаления.
    """
    from cli.support import get_file_paths  # Предполагается, что эта функция запрашивает пути к файлам
    type_obj, video_path, txt_path = get_file_paths()
    if not video_path and not txt_path:
        return "Ни один источник не выбран. Возврат в главное меню."
    
    # Добавление объекта
    add_message = RAG.add_object_from_files(assistant, video_path, txt_path)
    print(add_message)
    
    # Извлечение ID добавленного объекта из сообщения
    id_search = re.search(r"ID:\s*(\d+)", add_message)
    if not id_search:
        return "Не удалось извлечь ID добавленного объекта. Сообщение: " + add_message
    object_id = int(id_search.group(1))

    # Определяем тип добавленного объекта. Попробуем найти его в обоих списках.
    added_obj = RAG.get_object_by_id(object_id, "kandidate")
    if not added_obj:
        added_obj = RAG.get_object_by_id(object_id, "project")
    
    if not added_obj:
        return "Не удалось найти добавленный объект по ID: " + object_id
    
    current_type = (added_obj.get("type") or added_obj.get("Type") or "").lower().strip()
    if current_type in ["kandidate", "программист"]:
        delete_type = "kandidate"  # удаляем из списка кандидатов
    
    elif current_type in ["project", "проект"]:
        delete_type = "project"  # удаляем из списка проектов
    
    else:
        return "Неверный тип объекта: " + current_type

    # Выполнение подбора
    match_result = RAG.match_object(assistant, object_id, current_type)

    # Удаление объекта
    delete_result = RAG.delete_object(object_id, delete_type)
    print("Удаление объекта:")
    print(delete_result)
    
    final_message = (
        f"Операция завершена успешно.\n"
        f"ID объекта: {object_id}\n"
        f"Результат подбора:\n{match_result}\n"
        f"Результат удаления: {delete_result}"
    )
    
    return final_message
