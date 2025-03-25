import faiss
import os
import logging
import json
from typing import Optional


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaissDB:
    """
    Класс FaissDB предоставляет статические методы для работы с FAISS индексами:
      - Определяет базовую директорию для хранения индексов (на системном диске: tempDiscription/faissData).
      - Сохраняет, загружает и удаляет индексы.
      - Предоставляет методы инициализации (connect) и сохранения (disconnect) всех индексов.
    """
    # Базовая директория для хранения данных (индексов)
    BASE_DIR: str = os.path.join(os.path.abspath(os.sep), "tempDiscription", "faissData")
    os.makedirs(BASE_DIR, exist_ok=True)

    # Пути к файлам для двух индексов: кандидаты и проекты
    CANDIDATES_INDEX_FILE: str = os.path.join(BASE_DIR, "candidates_index.bin")
    PROJECTS_INDEX_FILE: str = os.path.join(BASE_DIR, "projects_index.bin")
    CANDIDATES_DATA_FILE: str = os.path.join(BASE_DIR, "candidates_data.txt")
    PROJECTS_DATA_FILE: str = os.path.join(BASE_DIR, "projects_data.txt")

    # Загруженные индексы будут храниться здесь (при инициализации)
    candidates_index: Optional[faiss.Index] = []
    projects_index: Optional[faiss.Index] = []
    candidates_data: Optional[faiss.Index] = []
    projects_data: Optional[faiss.Index] = []    


    @staticmethod
    def load_index(file_path) -> Optional[faiss.Index]:
        """
        Загружает FAISS индекс из указанного файла.
        Если файла не существует, возвращает None.
        """
        if os.path.exists(file_path):
            try:
                index = faiss.read_index(file_path)
                logger.info(f"Loaded index from {file_path}")
                return index
            
            except Exception as e:
                logger.error(f"Failed to load index from {file_path}: {e}")
                return None
        
        else:
            logger.info(f"Index file {file_path} does not exist.")
            return None


    @staticmethod
    def save_index(index: faiss.Index, file_path: str) -> None:
        """
        Сохраняет FAISS индекс в указанный файл.
        """
        try:
            faiss.write_index(index, file_path)
            logger.info(f"Index saved to {file_path}")
        
        except Exception as e: logger.error(f"Error saving index to {file_path}: {e}")


    @staticmethod
    def delete_index(file_path: str) -> None:
        """
        Удаляет файл с индексом, если он существует.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Index file {file_path} deleted successfully.")
           
            else: logger.warning(f"Index file {file_path} does not exist.")
        
        except Exception as e: logger.error(f"Error deleting index file {file_path}: {e}")


    @classmethod
    def initialize(cls):
        # Загрузка индексов
        if os.path.exists(cls.CANDIDATES_INDEX_FILE):
            cls.candidates_index = faiss.read_index(cls.CANDIDATES_INDEX_FILE)
            print(f"Загружен FAISS-индекс кандидатов из {cls.CANDIDATES_INDEX_FILE}")
        else:
            cls.candidates_index = None

        if os.path.exists(cls.PROJECTS_INDEX_FILE):
            cls.projects_index = faiss.read_index(cls.PROJECTS_INDEX_FILE)
            print(f"Загружен FAISS-индекс проектов из {cls.PROJECTS_INDEX_FILE}")
        else:
            cls.projects_index = None

        # Загрузка JSON-данных
        if os.path.exists(cls.CANDIDATES_DATA_FILE):
            with open(cls.CANDIDATES_DATA_FILE, "r", encoding="utf-8") as f:
                cls.candidates_data = json.load(f)
                print(f"Загружены данные кандидатов из {cls.CANDIDATES_DATA_FILE}")
        else:
            cls.candidates_data = []

        if os.path.exists(cls.PROJECTS_DATA_FILE):
            with open(cls.PROJECTS_DATA_FILE, "r", encoding="utf-8") as f:
                cls.projects_data = json.load(f)
                print(f"Загружены данные проектов из {cls.PROJECTS_DATA_FILE}")
        else:
            cls.projects_data = []


    @staticmethod
    def save_all() -> None:
        """
        Сохраняет текущие индексы и данные в файлы.
        """
        # Сохраняем индексы
        if FaissDB.candidates_index is not None:
            FaissDB.save_index(FaissDB.candidates_index, FaissDB.CANDIDATES_INDEX_FILE)

        if FaissDB.projects_index is not None:
            FaissDB.save_index(FaissDB.projects_index, FaissDB.PROJECTS_INDEX_FILE)

        # Сохраняем данные
        try:
            with open(FaissDB.CANDIDATES_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(FaissDB.candidates_data, f, ensure_ascii=False, indent=2)
            print(f"Сохранены данные кандидатов в {FaissDB.CANDIDATES_DATA_FILE}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных кандидатов: {e}")

        try:
            with open(FaissDB.PROJECTS_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(FaissDB.projects_data, f, ensure_ascii=False, indent=2)
            print(f"Сохранены данные проектов в {FaissDB.PROJECTS_DATA_FILE}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных проектов: {e}")
