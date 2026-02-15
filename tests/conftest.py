import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Configure l'environnement de test"""
    os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost:5432/rag_db"
    os.environ["SECRET_KEY"] = "test-secret-key"
    os.environ["ALGORITHM"] = "HS256"
    os.environ["EMBEDDING_MODEL"] = "BAAI/bge-base-en-v1.5"
