import os
import tempfile
import pytest

os.environ["FORGECARE_DATA"] = tempfile.mkdtemp(prefix="forgecare-pytest-")


@pytest.fixture(autouse=True)
def isolate_database():
    from forgecare.config import DB_PATH, KNOWLEDGE_DB
    for path in (DB_PATH, KNOWLEDGE_DB):
        path.unlink(missing_ok=True)
