import sys
import os
from PyQt6 import QtWidgets

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from razzy_rabbit.gui.gui_launcher import RazzyGuiLauncher


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = RazzyGuiLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
