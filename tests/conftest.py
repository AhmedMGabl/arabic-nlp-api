"""Shared pytest fixtures."""

from __future__ import annotations

import sys
import os

# Add project root to sys.path so imports resolve without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped synchronous test client (no async overhead for unit tests)."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---- Reusable Arabic text fixtures ----

@pytest.fixture
def positive_text() -> str:
    return "هذا المنتج رائع جداً وأنصح به بشدة، جودة ممتازة وخدمة مذهلة!"


@pytest.fixture
def negative_text() -> str:
    return "الخدمة سيئة جداً وكانت التجربة فظيعة ومحبطة للغاية"


@pytest.fixture
def neutral_text() -> str:
    return "الكتاب يحتوي على مئة وخمسين صفحة"


@pytest.fixture
def egyptian_text() -> str:
    return "إزيك؟ عاملين إيه يا جماعة؟ الأكل كان تمام أوي وكويس خالص"


@pytest.fixture
def gulf_text() -> str:
    return "شلونك؟ الحين وايد تعبان من الشغل، بس زين إنك جيت"


@pytest.fixture
def levantine_text() -> str:
    return "شو بدك؟ أنا هلق رح روح عالبيت، هيك أحسن"


@pytest.fixture
def diacritised_text() -> str:
    return "الرَّجُلُ يَقْرَأُ الكِتَابَ، وَيَفْهَمُ مَا فِيهِ!"


@pytest.fixture
def ner_text() -> str:
    return "أعلن الدكتور محمد العبدالله في الرياض عن إطلاق شركة التقنية الجديدة يوم الاثنين"
