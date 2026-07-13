from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LEAPS_OFFLINE_DATA_PATH", "/tmp/leaps-test-offline")
os.environ.setdefault("LEAPS_SKIP_ONBOARDING", "1")

import pytest


@pytest.fixture(scope="session")
def qapp():
    from leaps.app import create_application

    return create_application(["pytest"])
