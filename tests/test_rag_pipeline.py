import pytest
from app.rag.chunking import hybrid_chunking
from app.rag.embeddings import get_embeddings


import numpy as np
from unittest.mock import patch, MagicMock

@patch("app.rag.chunking.log_chunking_config")
@patch("app.rag.chunking.RECURSIVE_SPLITTER")
@patch("app.rag.chunking.SEMANTIC_SPLITTER")
def test_hybrid_chunking(
    mock_semantic_splitter,
    mock_recursive_splitter,
    mock_log_config,
):
    
    mock_semantic_splitter.split_text.return_value = [
        "Texte valide " * 10 
    ]

    mock_recursive_splitter.split_text.return_value = [
        "Sous chunk valide " * 10
    ]

   
    docs = [
        {
            "text": "Ceci est un document de test. " * 20,
            "metadata": {"source": "test_doc"}
        }
    ]

    
    chunks = hybrid_chunking(docs)

    
    assert isinstance(chunks, list)
    assert len(chunks) > 0

    first_chunk = chunks[0]

    assert "text" in first_chunk
    assert "metadata" in first_chunk

    assert first_chunk["metadata"]["doc_id"] == 0
    assert first_chunk["metadata"]["semantic_part"] == 0
    assert first_chunk["metadata"]["sub_part"] == 0

    mock_log_config.assert_called_once()




def test_embedding_model_loads():
    with patch("app.rag.embeddings.get_embeddings") as mock_get_model:
        fake_model = MagicMock()
        mock_get_model.return_value = fake_model

        model = get_embeddings()
        assert model is not None
        mock_get_model.assert_called_once()


def test_embedding_generation():
    with patch("app.rag.embeddings.get_embeddings") as mock_get_model:
        fake_model = MagicMock()

        fake_model.encode.return_value = np.zeros(768)
        mock_get_model.return_value = fake_model

        model = get_embeddings()
        embedding = model.encode("Test de maintenance biomédicale")

        assert embedding is not None
        assert len(embedding) == 768
        assert embedding.shape[0] == 768