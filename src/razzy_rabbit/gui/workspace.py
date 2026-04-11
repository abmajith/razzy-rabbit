import os
from PyQt6 import QtWidgets, QtCore
from pyvistaqt import QtInteractor
import pyvista as pv
from razzy_rabbit.core.fea_solver import FEASolver
import numpy as np

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
            box = pv.Box(bounds=(-5, 5, -5, 5, -0.5, 0.5))
            cyl = pv.Cylinder(center=(0, 0, 0), radius=2, height=2.0)
    
            # 1. Merge them into one object
            combined = box.merge(cyl)
    
            # 2. THE HEALING PROCESS:
            # 'clean' removes duplicate points
            # 'extract_surface' with 'dataset_surface' removes internal overlapping faces
            # 'triangulate' ensures GMSH only sees 3-sided polygons
            mesh = combined.clean().extract_surface(algorithm='dataset_surface').triangulate()

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

    
    def display_results(self, result_path):
        """Applies results specifically to the volume_mesh attribute."""
        
        # 1. Initialize solver (using the directory where results live)
        import os
        solver = FEASolver(os.path.dirname(result_path))
        
        # 2. Get the N,3 displacement array
        results = solver.parse_frd(result_path)
        u_data = results.get('U', [])
        
        # 3. Check the correct attribute: volume_mesh
        target_mesh = getattr(self, 'volume_mesh', None)
        
        if target_mesh is None:
            print("Error: No volume_mesh found in this workspace.")
            return

        if u_data.shape[0] != target_mesh.n_points:
            print(f"Re-aligning data: {u_data.shape[0]} results -> {target_mesh.n_points} mesh nodes")
            u_data = u_data[:target_mesh.n_points, :]

        # 4. Apply the data
        if len(u_data) == target_mesh.n_points:
            target_mesh.point_data["Displacement"] = u_data
            mag = np.linalg.norm(u_data, axis=1)
            target_mesh.point_data["Mag"] = mag
            
            # Warp it so we can see the deformation
            warped = target_mesh.warp_by_vector("Displacement", factor=1.5)
            
            # 5. Show it!
            self.plotter.add_mesh(
                warped, 
                scalars="Mag", 
                cmap="jet", 
                name="fea_results", 
                show_scalar_bar=True
            )
            self.plotter.reset_camera()
        else:
            print(f"Mismatch! Mesh Nodes: {target_mesh.n_points} vs Results: {len(u_data)}")
    
    @property
    def current_mesh(self):
        """Interface to get the most advanced mesh available."""
        if hasattr(self, 'volume_mesh') and self.volume_mesh:
            return self.volume_mesh
        return getattr(self, 'surface_mesh', None)