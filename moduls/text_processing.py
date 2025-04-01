import json
import logging
import os
from typing import Optional, Dict, Any


# Список обязательных ключей, которые должны присутствовать в итоговом словаре.
REQUIRED_KEYS = ["type", "name", "description", "stack", "skils", "telephone", "email", "telegram"]


# =====================================================================
# Подготовка promt
# =====================================================================
def prepare_summary_request(text: str) -> str:
    """
    Подготавливает запрос для ChatGPT на основе входного текста.

    Функция формирует текст запроса, в котором ChatGPT должно проанализировать входной текст собеседования и 
    вернуть результат в виде корректного JSON-объекта с обязательными ключами:
        - "Name" (ФИО кандидата или наименование проекта)
        - "Description" (краткое резюме кандидата или описание проекта)
        - "Stack" (перечень фреймворков и технологий)
        - "Skils" (ключевые навыки)
        - "telephone" (контактный телефон)
        - "email" (адрес электронной почты)
        - "telegram" (контакт в Telegram)

    Если какая-либо информация отсутствует, необходимо вернуть значение null.

    :param text: Исходный текст собеседования.
    :return: Запрос в виде строки, готовый для отправки в ChatGPT.
    """
    prompt = (
        "Анализируй предоставленный текст.\n"
        "Твоя задача — извлечь информацию и вернуть строго JSON-объект с ровно следующими ключами "
        "(все ключи должны быть в нижнем регистре):\n"
        "  \"name\", \"description\", \"stack\", \"skils\", \"telephone\", \"email\", \"telegram\".\n\n"
        "Правила заполнения:\n"
        "1. Поля \"stack\", \"skils\" и \"description\" должны содержать только сухую техническую информацию.\n"
        "   - \"description\": краткое, но полное описание проекта или опыта, сосредоточенное на технологических "
        "аспектах, включая годы работы, основные функции и цели.\n"
        "   - \"stack\": список технологий и инструментов, используемых в проекте или необходимых кандидату, "
        "включая все актуальные технологии.\n"
        "   - \"skils\": полное описание технических навыков и умений, относящихся к проекту или кандидату, "
        "включая все ключевые компетенции.\n"
        "2. Поле \"name\":\n"
        "   - Если текст описывает проект, значение должно быть названием проекта;\n"
        "   - Если текст относится к кандидату, значение должно быть именем кандидата.\n"
        "3. Поля \"telephone\", \"email\" и \"telegram\" должны содержать предоставленные данные "
        "или значение \"отсутствует\", если данных нет.\n"
        "4. Возвращай только JSON без каких-либо пояснений или дополнительного текста.\n\n"
        "Вот текст для анализа:\n"
    )

    return prompt


def call_chatgpt(assistant: Any, prompt: str) -> str:
    """
    Отправляет сформированный запрос в ChatGPT через объект ассистента и получает ответ.

    Предполагается, что объект assistant реализует метод send_message, принимающий строку и 
    возвращающий ответ от ChatGPT в виде строки.

    :param assistant: Объект, реализующий взаимодействие с ChatGPT.
    :param prompt: Подготовленный запрос для ChatGPT.
    :return: Ответ от ChatGPT в виде строки.
    """
    try:
        response = assistant.send_message(prompt)
        return response
    except Exception as e:
        logging.error("Ошибка при отправке запроса в ChatGPT: %s", e)
        raise


# =====================================================================
# Извлеякекаем словарь из ответа ChatGPT
# =====================================================================
def extract_json_from_text(text: str) -> Optional[str]:
    """
    Извлекает JSON-подстроку из общего текста.

    Функция ищет первый символ '{' и последний символ '}' в тексте и возвращает подстроку между ними,
    если они найдены. Если не удается найти корректную JSON-подстроку, возвращает None.

    :param text: Исходный текст, содержащий JSON и, возможно, лишний текст.
    :return: Подстрока, содержащая JSON, или None, если не найдено.
    """
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end+1]


def parse_chatgpt_response(response: str) -> Dict[str, Any]:
    """
    Парсит строку-ответ от ChatGPT в словарь.

    Функция сначала пытается напрямую преобразовать весь ответ в JSON. Если это не удается,
    она извлекает подстроку, содержащую JSON (по первому символу '{' и последнему '}' в ответе)
    и затем пытается распарсить её.

    :param response: Строка-ответ от ChatGPT.
    :return: Словарь с данными, извлеченными из ответа.
    :raises ValueError: Если не удается найти корректный JSON.
    """
    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        # Если прямой парсинг не удался, пытаемся извлечь JSON-подстроку
        json_str = extract_json_from_text(response)
        if json_str is None:
            logging.error("Не удалось найти JSON в ответе от ChatGPT")
            raise ValueError("Ответ от ChatGPT не содержит корректного JSON")
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            logging.error("Ошибка парсинга извлеченного JSON: %s", e)
            raise ValueError("Извлеченный JSON из ответа ChatGPT некорректен")


def normalize_summary_dict(summary: dict) -> dict:
    """
    Нормализует входной словарь, приводя все ключи к нижнему регистру и объединяя дублирующие варианты.
    Если обязательный ключ отсутствует или его значение пустое, устанавливает его значение как "Неизвестно".

    :param summary: Словарь, полученный после парсинга ответа ChatGPT.
    :return: Нормализованный словарь с ключами в нижнем регистре и заполненными обязательными полями.
    """
    # Создадим новый словарь с ключами в нижнем регистре
    normalized = {}
    for key, value in summary.items():
        lower_key = key.lower()
        # Если для одного ключа уже есть значение, оставляем первое непустое значение
        if lower_key in normalized:
            if not normalized[lower_key] and value:
                normalized[lower_key] = value
        else:
            normalized[lower_key] = value

    # Для каждого обязательного ключа, если он отсутствует или пустой, устанавливаем "Неизвестно"
    for req_key in REQUIRED_KEYS:
        if req_key not in normalized or not normalized[req_key]:
            normalized[req_key] = "Неизвестно"

    return normalized


def log_gpt_response(text: str, data: dict) -> None:
    log_path = os.path.join(os.getcwd(), "log_GPT_response.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("исходный текст:\n")
        f.write(f"{text}\n")
        f.write("Выходной словарь:\n")
        for key in data:
            f.write(f"{key} : {data[key]}\n")
        f.write("\n")


# =====================================================================
# Ключевой пайплайн анализа и парсинга текста
# =====================================================================
def process_text_summary(text: str, assistant: Any) -> Dict[str, Any]:
    """
    Основная функция для обработки текста с помощью ChatGPT.

    Функция выполняет следующие этапы:
      1. Подготовка запроса для ChatGPT с использованием входного текста.
      2. Отправка запроса в ChatGPT через объект ассистента.
      3. Получение ответа и его парсинг в словарь.
      4. Валидация итогового словаря на наличие всех необходимых ключей.
    
    :param text: Исходная строка с информацией, которую необходимо проанализировать.
    :param assistant: Объект ассистента, который осуществляет взаимодействие с ChatGPT.
    :return: Итоговый словарь с ключами: Type, Name, Description, Stack, Skils, Телефон, email, telegram.
    """
    # Шаг 1: Подготовка запроса для ChatGPT
    prompt = prepare_summary_request(text)
    logging.debug("Подготовленный запрос для ChatGPT: %s", prompt)
    
    # Шаг 2: Отправка запроса в ChatGPT через ассистента
    response = call_chatgpt(assistant, prompt)
    logging.debug("Получен ответ от ChatGPT: %s", response)
    
    # Шаг 3: Парсинг ответа в словарь (с предварительным выделением JSON, если необходимо)
    summary_dict = parse_chatgpt_response(response)
    logging.debug("Распарсенный словарь: %s", summary_dict)
    
    # Шаг 4: Валидация итогового словаря - добавление недостающих ключей со значением None
    summary_dict = normalize_summary_dict(summary_dict)
    logging.debug("Валидированный итоговый словарь: %s", summary_dict)
    log_gpt_response(text, summary_dict)

    return summary_dict
