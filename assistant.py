import openai
import time
import os

#Берем ключ из переменной окржуения.
API_KEY = os.getenv('OPENAI_API_KEY') 
ASSISTANT_ID = None

if not API_KEY:
    raise ValueError("OPENAI_API_KEY не найден. Добавьте его.")

def create_assistant(description):
    """
    Создает нового ассистента с указанным описанием.
    """
    try:
        client = openai.Client(api_key=API_KEY)
        assistant = client.beta.assistants.create(
            name="Job Interview Analyst",
            instructions=description,
            model="gpt-4o"
        )
        print("Создан ассистент:", assistant)
        return assistant.id
    except Exception as e:
        print("Ошибка при создании ассистента:", str(e))
        return None

if ASSISTANT_ID is None:
    description = """
        Ты ассистент для анализа собеседований с программистами. 
        Твоя задача - проанализировать текст интервью и резюме кандидата.
        Выдели основные навыки кандидата, его сильные и слабые стороны.
        Представь результат в виде краткого саммари в пару абзацев.
    """
    ASSISTANT_ID = create_assistant(description)

def generate_response(assistant_id, prompt, thread_id=None):
    """
    Генерирует ответ на основе предоставленного prompt.
    Создает новый чат, если thread_id не указан.
    """
    try:
        client = openai.Client(api_key=API_KEY)
        assistant = client.beta.assistants.retrieve(assistant_id)
        
        if thread_id is None:
            thread = client.beta.threads.create(messages=[{"role": "user", "content": prompt}])
        else:
            thread = client.beta.threads.retrieve(thread_id)
            client.beta.threads.messages.create(thread_id=thread.id, role='user', content=prompt)

        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)

        # Ожидаем завершения выполнения запроса
        while run.status != 'completed':
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            print(run.status)
            time.sleep(5)

        thread_messages = client.beta.threads.messages.list(thread.id)
        if thread_messages.data:
            return thread_messages.data[0].content[0].text.value, thread.id
        else:
            print("Ошибка: Нет сообщений в потоке.")
            return None
    except Exception as e:
        print("Ошибка при запросе:", str(e))
        return None

def analyze_candidate(assistant_id, interview_text):
    """
    Анализирует кандидата на основе текста интервью.
    """
    prompt = f"""
    Проанализируй пожалуйста этого кандидата на основе текста его собеседования.
    Текст собеседования: 
    {interview_text}

    Сделай выводы о его знаниях, в чем он разбирается лучше, в чем хуже и в какой области программирования он мог бы лучше показать себя.
    Сделай краткое саммари по кандидату.
    Напиши в такой структуре: Плюсы, минусы, в какой области программирования он мог бы лучше показать себя.
    """
    
    response = generate_response(assistant_id, prompt)
    if response is None:
        return "Ошибка при генерации ответа."
    
    answer, _ = response
    return answer

def save_to_file(save_folder, file_name, content):
    """
    Сохраняет ответ в файл в указанной папке.
    """
    try:
        # Проверяем, указана ли папка, если нет — используем текущую директорию
        if not save_folder:
            save_folder = os.getcwd()  # Текущая папка

        # Создаем папку, если её нет
        os.makedirs(save_folder, exist_ok=True)

        file_path = os.path.join(save_folder, file_name)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"Ответ сохранен в файл: {file_path}")
    except Exception as e:
        print(f"Ошибка при сохранении в файл: {str(e)}")

def start_assistant(text_file_path, save_folder=None):
    """
    Сбор саммари по кандидату и сохранение результата в файл.
    """
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            interview_text = f.read()
    except FileNotFoundError:
        print("Ошибка: Файл не найден.")
        return

    analysis_summary = analyze_candidate(ASSISTANT_ID, interview_text)
    
    if analysis_summary:
        # Создаем имя для файла
        base_filename = os.path.splitext(os.path.basename(text_file_path))[0]
        summary_filename = f"{base_filename}_ai_summary.txt"
        
        # Сохраняем результат в указанную папку
        save_to_file(save_folder, summary_filename, analysis_summary)
    else:
        print("Не удалось получить анализ")
