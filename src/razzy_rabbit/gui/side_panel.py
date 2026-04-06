from PyQt6 import QtWidgets, QtCore

class RazzySidePanel(QtWidgets.QDockWidget):
    def __init__(self, main_window):
        super().__init__("Project Explorer", main_window)
        self.main_win = main_window
        
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        
        # Layer Controls
        group = QtWidgets.QGroupBox("Visibility")
        g_layout = QtWidgets.QVBoxLayout()
        
        self.chk_surface = QtWidgets.QCheckBox("Surface (STL)")
        self.chk_surface.setChecked(True)
        self.chk_surface.stateChanged.connect(self.main_win.update_layer_visibility)
        
        self.chk_mesh = QtWidgets.QCheckBox("Volume Mesh")
        self.chk_mesh.setChecked(True)
        self.chk_mesh.stateChanged.connect(self.main_win.update_layer_visibility)
        
        g_layout.addWidget(self.chk_surface)
        g_layout.addWidget(self.chk_mesh)
        group.setLayout(g_layout)
        
        layout.addWidget(group)
        layout.addStretch()
        self.setWidget(container)
