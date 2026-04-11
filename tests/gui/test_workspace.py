import pytest
from PyQt6.QtWidgets import QApplication
from razzy_rabbit.gui.workspace import RazzyWorkspace

def test_workspace_initialization(qtbot):
    """Ensures the 3D workspace opens without errors."""
    window = RazzyWorkspace()
    qtbot.addWidget(window)
    
    assert window.plotter is not None
    assert window.isVisible()
