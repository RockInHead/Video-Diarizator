import os
from openai import OpenAI
import logging
from dotenv import load_dotenv


class GPTAssistant:
    """
    Класс GPTAssistant реализует подключение к API OpenAI и предоставляет метод отправки сообщений.
    
    При инициализации класс:
      - Загружает API-ключ из переменных окружения (например, OPENAI_API_KEY).
      - Устанавливает ключ для библиотеки openai.
      
    Метод send_message отправляет сообщение в ChatGPT и возвращает ответ в виде строки.
    """
    load_dotenv() 
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def __init__(self):
        """
        Инициализирует объект GPTAssistant.
        
        Загружает API-ключ из переменной окружения OPENAI_API_KEY и настраивает openai.
        Если ключ не найден, выбрасывает исключение.
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key: raise ValueError("OPENAI_API_KEY не найден в переменных окружения")
        logging.info("GPTAssistant инициализирован с использованием API-ключа.")

    def send_message(self, message: str) -> str:
        """
        Отправляет сообщение в ChatGPT и возвращает ответ в виде строки.
        
        Использует OpenAI ChatCompletion API (модель gpt-3.5-turbo).
        Метод формирует системное сообщение (можно изменить или расширить) и сообщение пользователя.
        
        :param message: Строка с запросом для ChatGPT.
        :return: Строка с ответом от ChatGPT.
        :raises Exception: При ошибке запроса выбрасывается исключение.
        """
        try:
            response = GPTAssistant.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты ассистент, помогающий анализировать и структурировать текст."},
                    {"role": "user", "content": message}
                ])
            # Извлекаем текст ответа из полученного объекта
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            logging.error("Ошибка при отправке сообщения в ChatGPT: %s", e)
            raise
