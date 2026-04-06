from PyQt6 import QtWidgets, QtCore  # Import QtCore directly here
from razzy_rabbit.gui.workspace import RazzyWorkspace
from razzy_rabbit.gui.toolbar import RazzyToolbar
from razzy_rabbit.gui.side_panel import RazzySidePanel
from razzy_rabbit.core.mesh_engine import generate_tetra_mesh

class RazzyGuiLauncher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Razzy-Rabbit Suite")
        self.resize(1200, 800)

        # 1. Setup Central Workspace Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.setCentralWidget(self.tabs)

        # 2. Add Dismantled Components
        self.toolbar = RazzyToolbar(self)
        self.addToolBar(self.toolbar)
        
        self.side_panel = RazzySidePanel(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.side_panel)
        self.add_new_workspace("Design 1")

    def get_current_ws(self) -> RazzyWorkspace:
        return self.tabs.currentWidget()

    def add_new_workspace(self, name="New Project"):
        ws = RazzyWorkspace()
        self.tabs.addTab(ws, name)
        return ws
    
    def trigger_selection(self, mode):
        ws = self.get_current_ws()
        if ws:
            ws.enable_selection(mode)

    # --- Communication Bridge ---
    def trigger_meshing(self):
        ws = self.get_current_ws()
        if ws and ws.surface_mesh:
            vol_mesh = generate_tetra_mesh(ws.surface_mesh)
            ws.add_volume_mesh(vol_mesh)

    def update_layer_visibility(self):
        ws = self.get_current_ws()
        if ws:
            ws.set_layer_visibility("surface", self.side_panel.chk_surface.isChecked())
            ws.set_layer_visibility("mesh", self.side_panel.chk_mesh.isChecked())

    def load_file_trigger(self):
        ws = self.get_current_ws()
        if ws: ws.request_load_stl() # Workspace handles its own file dialog

    def gen_sample_trigger(self, p_type):
        ws = self.get_current_ws()
        if ws: ws.generate_sample(p_type)

    def _close_tab(self, index):
        self.tabs.widget(index).plotter.close()
        self.tabs.removeTab(index)
