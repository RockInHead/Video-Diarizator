import logging
import time
from typing import Dict, Any, Tuple
from core.storage.faiss_controller import add_document, delete_object, search_object
from core.storage.faiss_db import FaissDB

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RAG:
    """
    Класс RAG – единый интерфейс для работы с RAG-системой.
    Предоставляет статические методы для:
      1) Добавления объекта в FAISS (полный пайплайн от обработки файлов до индексирования),
      2) Подбора обратного типа объектов по ID,
      3) Получения списка всех объектов заданного типа,
      4) Удаления объекта по ID,
      5) Получения объекта по ID.
      
    Предполагается, что данные уже корректно обработаны (например, нормализация ключей произведена в text_processing).
    """

    @staticmethod
    def add_object(data: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Добавляет объект в FAISS.
        Генерирует уникальный ID (если отсутствует) и вызывает функцию add_document для индексирования.
        Логирует данные перед добавлением (чтобы можно было проверить, как сформирован Document).
        
        :param data: Словарь с данными объекта (ожидается, что ключи приведены к единому регистру, например, "type", "name", "description", "stack", "skils", "telephone", "email", "telegram").
        :return: (True, id) при успехе или (False, error_message) при ошибке.
        """
        try:
            if "id" not in data or not data["id"]:
                data["id"] = int(time.time() * 1000)
            logger.info("Добавление объекта")
            add_document(data)
            logger.info("Объект успешно добавлен, id: %s", data["id"])
            return (True, data["id"])
        
        except Exception as e:
            logger.error("Ошибка при добавлении объекта: %s", e)
            return (False, str(e))

    @staticmethod
    def add_object_from_files(assistant: Any, video_path: str, txt_path: str) -> str:
        """
        Полный пайплайн добавления объекта:
          1. Если ни видео, ни текст не указаны, возвращает ошибку.
          2. Если задан видеофайл, вызывает extraction_text из moduls.video_processing для извлечения текста.
          3. Если задан текстовый файл, считывает его содержимое.
          4. Объединяет полученные тексты.
          5. Вызывает process_text_summary (из moduls.text_processing) для получения структурированного словаря.
          6. Вызывает add_object для индексирования.
          7. Возвращает итоговое сообщение с результатом.
          
        :param assistant: Объект GPTAssistant.
        :param video_path: Путь к видеофайлу (может быть None).
        :param txt_path: Путь к текстовому файлу (может быть None).
        :return: Итоговое сообщение.
        """
        if not video_path and not txt_path:
            return "Ошибка: необходимо указать видео или текстовое описание."
        
        extracted_text = ""
        
        if video_path:
            print("Начата процедура извлечения текста из видео...")
            try:
                from moduls.video_processing import extraction_text
                video_text = extraction_text(video_path)
                print("Текст из видео получен.")
                extracted_text += video_text + "\n"

            except Exception as e:
                return f"Ошибка при извлечении текста из видео: {e}"
        
        if txt_path:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    file_text = f.read().strip()

                if file_text:
                    print("Данные из файла получены.")
                    extracted_text += file_text + "\n"

                else:
                    print("Текстовый файл пустой.")

            except Exception as e:
                return f"Ошибка при чтении текстового файла: {e}"
        
        if not extracted_text.strip(): return "Ошибка: итоговый текст пустой."
        
        try:
            from moduls.text_processing import process_text_summary
            parsed_dict = process_text_summary(extracted_text, assistant)
            print(f"Словарь после парсинга:\n{parsed_dict}")
            if not parsed_dict:
                return "Ошибка: парсер вернул пустой результат."
            
        except Exception as e: return f"Ошибка при обработке текста: {e}"
        
        success, result = RAG.add_object(parsed_dict)
        if not success: return f"Ошибка при добавлении объекта: {result}"
        
        obj_id = result
        print(f"Новый объект добавлен, ID: {obj_id}")
        return f"Объект успешно добавлен, ID: {obj_id}"
    
    @staticmethod
    def generate_match_prompt(source_obj: dict, results: list) -> str:
        """
        Формирует prompt для генерации итогового текста подбора объектов.
        """
        source_type = source_obj.get("type", "").lower().strip()
        source_name = source_obj.get("name", "Неизвестно")
        source_stack = source_obj.get("stack", "отсутствует")
        source_skils = source_obj.get("skils", "отсутствует")

        if source_type == "kandidate":
            source_label = "кандидата"
            result_label = "проекты"
        elif source_type == "project":
            source_label = "проекта"
            result_label = "кандидаты"
        else:
            source_label = "объекта"
            result_label = "результаты"

        cards = []
        for idx, res in enumerate(results, start=1):
            meta = res.get("metadata", {})
            card = (
                f"   {idx}. {meta.get('name', 'Неизвестно')}\n"
                f"   Stack: {meta.get('stack', 'отсутствует')}\n"
                f"   Skils: {meta.get('skils', 'отсутствуют')}\n"
                f"   Description: {meta.get('description', 'отсутствует')}\n"
                f"   Телефон: {meta.get('telephone', 'отсутствует')}\n"
                f"   Email: {meta.get('email', 'отсутствует')}\n"
                f"   Telegram: {meta.get('telegram', 'отсутствует')}"
            )
            cards.append(card)

        cards_text = "\n".join(cards)

        prompt = (
            f"Ты — эксперт по подбору {result_label}. У тебя есть данные {source_label} \"{source_name}\" "
            f"со стеком: {source_stack};\n "
            f"и скилами {source_skils}.\n\n"
            f"Вот список подходящих {result_label}:\n{cards_text}\n\n"
            f"Сформируй краткий и человечный ответ, который будет выглядеть так: "
            f"\"Для {source_label} {source_name} подходят следующие {result_label}:\n"
            f"1. [Имя из объекта списка {cards_text}] со стеком [Стек из объекта списка {cards_text}]. "
            f"[Краткое объяснение, почему один из {result_label} подходит {source_label}].  \n"
            f"Контакты: [Контакты из объекта списка {cards_text}].\n\""
            f"Далее в таком же формате перечисляй все остальные {result_label} из списка, если они есть. \n"
            f"Используй более разговорный стиль, указывая название/имя, стек, скилы, "
            f"и краткое объяснение, без лишних комментариев. "
            f"Учти, что ответ пишешь беспристрастному лицу, осуществляющему подбор"
        )
        return prompt

    @staticmethod
    def match_object(assistant: Any, object_id: Any, doc_type: str) -> str:
        """
        Подбирает обратный тип объектов для заданного объекта по его ID.
        Формирует запрос на основе полей "stack", "skils" и "description",
        затем генерирует prompt для ChatGPT и отправляет его через assistant.send_message.
        
        :param assistant: Объект GPTAssistant.
        :param object_id: ID объекта, по которому осуществляется подбор.
        :param doc_type: Оригинальный тип объекта ("kandidate" или "project").
        :return: Форматированный результат подбора или сообщение об ошибке.
        """
        try:
            source_obj = RAG.get_object_by_id(object_id, doc_type)
            if not source_obj:
                return f"Объект с id {object_id} не найден."
            
            # Определяем обратный тип для подбора
            if doc_type in ["kandidate", "программист"]:
                reverse_type = "project"

            elif doc_type in ["project", "проект"]:
                reverse_type = "kandidate"

            else:
                return "Invalid document type."
            
            query = {
                "type": reverse_type,
                "stack": source_obj.get("stack", ""),
                "skils": source_obj.get("skils", ""),
                "description": source_obj.get("description", "")
            }
            results = search_object(query, top_k=5)
            prompt = RAG.generate_match_prompt(source_obj, results)
            formatted_message = assistant.send_message(prompt)
            return "\n\n" + formatted_message
        
        except Exception as e:
            logger.error(f"Error in match_object: {e}")
            return f"Error in match_object: {e}"

    @staticmethod
    def get_all_objects(doc_type: str) -> str:
        """
        Возвращает список всех объектов заданного типа в виде форматированной строки.
        
        :param doc_type: "kandidate" или "project".
        :return: Форматированная строка с объектами или сообщение об ошибке.
        """
        try:
            if doc_type not in ["kandidate", "project"]: return "Invalid document type."
            data_list = FaissDB.candidates_data if doc_type == "kandidate" else FaissDB.projects_data
            
            if not data_list: return f"No objects found for type {doc_type}."
            output = f"Список всех {'кандидатов' if doc_type=='kandidate' else 'проектов'}:\n"
            for idx, obj in enumerate(data_list, start=1):
                name = obj.get("name", "Неизвестно")
                stack = obj.get("stack", "")
                output += f"{idx} - {name}\nСтэк: {stack}\n\n"
            
            return output
        
        except Exception as e:
            logger.error(f"Error in get_all_objects: {e}")
            return f"Error in get_all_objects: {e}"

    @staticmethod
    def delete_object(object_id: Any, doc_type: str) -> str:
        """
        Удаляет объект по его ID.
        
        :param object_id: ID объекта.
        :param doc_type: "kandidate" или "project".
        :return: Сообщение об успехе или ошибка.
        """
        try:
            if not object_id:
                return "Invalid id."
            
            if doc_type not in ["kandidate", "project"]:
                return "Invalid document type."
            
            delete_object(object_id, doc_type)
            return f"Object with id {object_id} deleted successfully from {doc_type} database."
        
        except Exception as e:
            logger.error(f"Error in delete_object: {e}")
            return f"Error in delete_object: {e}"

    @staticmethod
    def get_object_by_id(object_id: Any, doc_type: str) -> dict:
        """
        Возвращает объект по его ID.
        
        :param object_id: ID объекта.
        :param doc_type: "kandidate" или "project".
        :return: Словарь с данными объекта или пустой словарь, если объект не найден.
        """
        try:
            if doc_type == "kandidate":
                for obj in FaissDB.candidates_data:
                    if obj.get("id") == object_id: return obj
            
            elif doc_type == "project":
                for obj in FaissDB.projects_data:
                    if obj.get("id") == object_id:
                        return obj
            return {}
        
        except Exception as e:
            logger.error(f"Error in get_object_by_id: {e}")
            return {}
