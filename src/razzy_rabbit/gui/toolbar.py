from PyQt6 import QtWidgets, QtGui

class RazzyToolbar(QtWidgets.QToolBar):
    def __init__(self, main_window):
        super().__init__("Main Controls", main_window)
        self.main_win = main_window
        self._setup_file_menu()
        self._setup_analysis_menu()

    def _setup_file_menu(self):
        btn = QtWidgets.QToolButton()
        btn.setText("📁 Workspace")
        btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(btn)
        
        menu.addAction("New Tab", self.main_win.add_new_workspace)
        menu.addAction("Import STL", self.main_win.load_file_trigger)
        
        samples = menu.addMenu("🛠 Samples")
        samples.addAction("Bracket", lambda: self.main_win.gen_sample_trigger("bracket"))
        samples.addAction("Link", lambda: self.main_win.gen_sample_trigger("link"))
        
        btn.setMenu(menu)
        self.addWidget(btn)

    def _setup_analysis_menu(self):
        btn = QtWidgets.QToolButton()
        btn.setText("🔬 Analysis")
        btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(btn)
        
        menu.addAction("Generate Mesh (GMSH)", self.main_win.trigger_meshing)
        menu.addSeparator()
        menu.addAction("📍 Mark Fixed Support", lambda: self.main_win.trigger_selection("fixed"))
        menu.addAction("⚖️ Apply Force/Load", lambda: self.main_win.trigger_selection("load"))
        btn.setMenu(menu)
        self.addWidget(btn)

        menu.addSeparator()
        solve_action = menu.addAction("🚀 Run CalculiX Solve")
        solve_action.triggered.connect(self.main_win.run_fea)
