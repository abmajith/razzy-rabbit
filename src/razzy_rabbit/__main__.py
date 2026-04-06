import warnings
import os
import sys 
from PyQt6.QtWidgets import QApplication
from razzy_rabbit.gui.gui_launcher import RazzyGuiLauncher


warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["VTK_SILENT"] = "1"

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Razzy-Rabbit Robotics")
    
    window = RazzyGuiLauncher()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
