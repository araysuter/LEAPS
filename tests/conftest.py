from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LEAPS_OFFLINE_DATA_PATH", "/tmp/leaps-test-offline")
os.environ.setdefault("LEAPS_SKIP_ONBOARDING", "1")
# Existing scientific fixtures exercise the portable CPU path. GPU tests opt in.
os.environ.setdefault("LEAPS_ALIGNMENT_DEVICE", "cpu")

import pytest


@pytest.fixture(scope="session")
def qapp():
    from leaps.app import create_application

    return create_application(["pytest"])
