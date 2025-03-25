import logging
import numpy as np
from typing import Dict, Any, List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from core.storage.faiss_db import FaissDB


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Инициализация эмбеддингов: используем модель "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    logger.info("HuggingFace embeddings initialized successfully.")

except Exception as e:
    logger.error(f"Error initializing embeddings: {e}")
    raise


def _prepare_embedding_text(data: Dict[str, Any]) -> str:
    """
    Объединяет поля 'stack', 'skils' и 'description' в одну строку для генерации эмбеддинга.
    Если какое-либо поле отсутствует, используется пустая строка.
    """
    stack = data.get("stack", "")
    skils = data.get("skils", "")
    description = data.get("description", "")
    return f"{stack} {skils} {description}".strip()


def _prepare_document(data: Dict[str, Any]) -> Document:
    """
    Преобразует словарь с данными в объект Document.
    Поле для индексирования формируется из объединения полей 'stack', 'skils', 'description'.
    Остальные поля сохраняются в metadata.
    """
    text = _prepare_embedding_text(data)
    metadata = data.copy()
    return Document(page_content=text, metadata=metadata)


def build_index(data_list: List[Dict[str, Any]]) -> FAISS:
    """
    Строит новый FAISS индекс на основе списка словарей.
    Использует метод FAISS.from_documents из LangChain для создания индекса.
    """
    documents = [_prepare_document(item) for item in data_list]
    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info("FAISS index built successfully with %d documents.", len(documents))
        return vector_store
    
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        raise


def add_document(data: Dict[str, Any]) -> None:
    doc_type = (data.get("type") or data.get("Type") or "").lower().strip()
    if doc_type not in ["программист", "kandidate", "проект", "project"]:
        raise ValueError("Invalid document type")
    
    if doc_type in ["программист", "kandidate"]:
        FaissDB.candidates_data.append(data)
        vector_store = build_index(FaissDB.candidates_data)
        FaissDB.candidates_index = vector_store.index
        FaissDB.save_index(FaissDB.candidates_index, FaissDB.CANDIDATES_INDEX_FILE)
    
    else:
        FaissDB.projects_data.append(data)
        vector_store = build_index(FaissDB.projects_data)
        FaissDB.projects_index = vector_store.index
        FaissDB.save_index(FaissDB.projects_index, FaissDB.PROJECTS_INDEX_FILE)

    FaissDB.save_all()



def delete_object(doc_id: Any, doc_type: str) -> None:
    """
    Удаляет объект по его id из глобального списка и пересоздаёт индекс.
    Обновлённый индекс сохраняется через FaissDB.
    """
    # Приводим doc_type к нижнему регистру
    doc_type = doc_type.lower().strip()
    if doc_type in ["программист", "kandidate"]:
        new_list = [doc for doc in FaissDB.candidates_data if doc.get("id") != doc_id]
        if len(new_list) == len(FaissDB.candidates_data):
            raise ValueError(f"No candidate found with id {doc_id}")
        
        FaissDB.candidates_data = new_list
        vector_store = build_index(FaissDB.candidates_data)
        FaissDB.candidates_index = vector_store.index
        FaissDB.save_index(FaissDB.candidates_index, FaissDB.CANDIDATES_INDEX_FILE)
    
    elif doc_type in ["проект", "project"]:
        new_list = [doc for doc in FaissDB.projects_data if doc.get("id") != doc_id]
        if len(new_list) == len(FaissDB.projects_data):
            raise ValueError(f"No project found with id {doc_id}")
        
        FaissDB.projects_data = new_list
        vector_store = build_index(FaissDB.projects_data)
        FaissDB.projects_index = vector_store.index
        FaissDB.save_index(FaissDB.projects_index, FaissDB.PROJECTS_INDEX_FILE)
    else:
        raise ValueError("Invalid document type for deletion")


def search_object(query_data: Dict[str, Any], top_k = None, threshold: float = 0.6) -> List[Dict[str, Any]]:
    query_type = (query_data.get("type") or query_data.get("Type") or "").lower().strip()
    if query_type not in ["программист", "kandidate", "проект", "project"]:
        raise ValueError("Invalid query type")
    
    query_text = _prepare_embedding_text(query_data)
    embedding = np.array([embeddings.embed_query(query_text)])

    if query_type in ["project", "проект"]:
        data_list = FaissDB.projects_data
        index = FaissDB.projects_index
    else:
        data_list = FaissDB.candidates_data
        index = FaissDB.candidates_index

    if not data_list:
        return []

    if index is None:
        raise ValueError("FAISS index is not initialized")

    try:
        # Запрашиваем столько, сколько объектов есть в базе
        distances, indices = index.search(embedding, len(data_list))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or dist < threshold:
                continue  # Пропуск нерелевантных

            try:
                metadata = data_list[idx]
                results.append({
                    "page_content": _prepare_embedding_text(metadata),
                    "metadata": metadata,
                    "similarity": dist
                })
            except IndexError:
                continue

        return results

    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise
