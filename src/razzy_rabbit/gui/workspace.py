from PyQt6 import QtWidgets, QtCore
from pyvistaqt import QtInteractor
import pyvista as pv

class RazzyWorkspace(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        
        # Local Data
        self.surface_mesh = None
        self.volume_mesh = None
        self.actors = {"surface": None, "mesh": None, "selection": None}
        self.selected_faces = []

        self.plotter.set_background("#0f0f0f")
        self.plotter.add_axes()
    
    
    def enable_selection(self, mode="fixed"):
        self.current_mode = mode

        # 1. Kill PyVista's default 'P' and 'B' behaviors
        self.plotter.clear_events_for_key('p')
        self.plotter.clear_events_for_key('b')
        self.plotter.disable_picking()

        # 2. Enable Point Picking with a "Left Click" trigger
        self.plotter.enable_point_picking(
            callback=self._handle_pick_event,
            left_clicking=True,
            show_message=True,
            show_point=False,
            message=f"MODE: {mode.upper()}\nLeft-Click to Add | Right-Click to Remove"
        )

    
    def _handle_pick_event(self, point):
        """Precision logic for tight mechanical meshes."""
        if point is None: return

        # 1. Much tighter threshold for unselecting (Precision Mode)
        # 0.05 means you have to be almost exactly on the marker to delete it
        existing_idx = self._find_nearby_point(point, threshold=0.05)

        if existing_idx is not None:
            # UNSELECT
            removed = self.selected_faces.pop(existing_idx)
            self.plotter.remove_actor(removed["marker_name"])
            print("Unselected point.")
        else:
            # SELECT
            marker_name = f"marker_{QtCore.QDateTime.currentMSecsSinceEpoch()}"
            self.selected_faces.append({
                "coord": point, 
                "mode": self.current_mode,
                "marker_name": marker_name
            })
            
            # 2. Ultra-small Cube (2% of a unit)
            marker_geom = pv.Cube(center=point, x_length=0.02, y_length=0.02, z_length=0.02)
            
            self.plotter.add_mesh(
                marker_geom,
                color="red" if self.current_mode == "fixed" else "green",
                name=marker_name,
                pickable=False,
                reset_camera=False # Essential: don't jump the camera
            )
            print(f"Precision point added at {point}")

    def _find_nearby_point(self, new_point, threshold=0.05):
        """Helper for precision detection."""
        import numpy as np
        for i, entry in enumerate(self.selected_faces):
            dist = np.linalg.norm(np.array(entry["coord"]) - np.array(new_point))
            if dist < threshold:
                return i
        return None

    def _find_nearby_point(self, new_point, threshold=0.3):
        """Helper to find if a click is 'close enough' to an existing marker to delete it."""
        import numpy as np
        for i, entry in enumerate(self.selected_faces):
            dist = np.linalg.norm(np.array(entry["coord"]) - np.array(new_point))
            if dist < threshold:
                return i
        return None

    def generate_sample(self, p_type):
        """Creates mechanical primitives directly in this workspace."""
        if p_type == "bracket":
            mesh = pv.Box(bounds=(-5, 5, -5, 5, -0.5, 0.5)).merge(pv.Cylinder(center=(0,0,1), radius=2))
        elif p_type == "link":
            mesh = pv.Cylinder(center=(0,0,0), direction=(1,0,0), radius=1, height=10)
        else: # mount
            mesh = pv.Box(bounds=(-2, 2, -2, 2, -2, 2)).clip('z')

        self.add_surface(mesh)

    def add_surface(self, mesh):
        self.surface_mesh = mesh
        if self.actors["surface"]:
            self.plotter.remove_actor(self.actors["surface"])
        self.actors["surface"] = self.plotter.add_mesh(
            mesh,
            color="lightgray",
            lighting=True,
            # This creates a uniform "Engineering" shading
            # without the distracting mirror reflections
            ambient=0.2,
            diffuse=0.8,
            specular=0.1,
            name="surface"
        )
        self.plotter.reset_camera()

    def add_volume_mesh(self, vol_mesh):
        self.volume_mesh = vol_mesh
        if self.actors["mesh"]:
            self.plotter.remove_actor(self.actors["mesh"])
        self.actors["mesh"] = self.plotter.add_mesh(
            vol_mesh, style="wireframe", color="yellow", 
            line_width=1, name="vol_mesh"
        )

    def set_layer_visibility(self, layer, visible):
        if self.actors.get(layer):
            self.actors[layer].SetVisibility(visible)
            self.plotter.render()
            
    def request_load_stl(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl)")
        if path:
            mesh = pv.read(path)
            self.add_surface(mesh)
