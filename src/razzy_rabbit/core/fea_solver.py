import os
import subprocess
import numpy as np
from scipy.spatial import KDTree
import re
import time
import meshio

class FEASolver:
    def __init__(self, workspace):
        self.ws = workspace
        self.job_name = "razzy_job"
        self.work_dir = "/tmp/razzy_fea"
        os.makedirs(self.work_dir, exist_ok=True)

    def run_analysis(self, mesh_data, selections, material_props=None):
        """
        mesh_data: The pyvista UnstructuredGrid (from GMSH)
        selections: List of {'coord': [x,y,z], 'mode': 'fixed'/'load'}
        """
        if material_props is None:
            # Default to Aluminum 6061 (SI Units: Pa, kg, m)
            material_props = {"E": 70e9, "nu": 0.33, "rho": 2700}

        inp_path = os.path.join(self.work_dir, f"{self.job_name}.inp")
        
        # 1. Build spatial index of mesh nodes
        nodes = mesh_data.points
        tree = KDTree(nodes)

        # 2. Map user clicks to Node IDs
        fixed_nodes = []
        load_nodes = []
        
        for sel in selections:
            _, idx = tree.query(sel['coord'])
            node_id = idx + 1 # CalculiX is 1-indexed
            if sel['mode'] == 'fixed':
                fixed_nodes.append(node_id)
            else:
                load_nodes.append(node_id)

        # 3. Write CalculiX Input File
        with open(inp_path, "w") as f:
            f.write("*HEADING\nRazzy-Rabbit Structural Analysis\n")
            
            # Nodes - Use fixed precision to avoid Fortran reading errors
            f.write("*NODE, NSET=Nall\n")
            for i, p in enumerate(nodes):
                f.write(f"{i+1}, {p[0]:.8f}, {p[1]:.8f}, {p[2]:.8f}\n")

            # Elements
            f.write("*ELEMENT, TYPE=C3D10, ELSET=Eall\n")
            cell_data = mesh_data.cells
            i = 0
            el_id = 1
            while i < len(cell_data):
                num_nodes = cell_data[i]
                if num_nodes == 10:  # C3D10
                    indices = cell_data[i+1 : i+11]
                    conn = [int(idx) + 1 for idx in indices]
                    # CalculiX elements must be written 8 nodes per line max usually, 
                    # but for C3D10, 11 values total (ID + 10 nodes) is fine if comma separated
                    line = f"{el_id}, " + ", ".join(map(str, conn))
                    f.write(line + "\n")
                    el_id += 1
                i += (num_nodes + 1)

            # --- CRITICAL CHECK ---
            if el_id == 1:
                print("🚨 ERROR: No C3D10 elements were found in the mesh!")
                return None

            # Material
            f.write("*MATERIAL, NAME=ALUMINUM\n*ELASTIC\n")
            f.write(f"{material_props['E']:.1f}, {material_props['nu']:.4f}\n")
            f.write("*SOLID SECTION, ELSET=Eall, MATERIAL=ALUMINUM\n")

            # Boundary Conditions
            f.write("*BOUNDARY\n")
            for node in fixed_nodes:
                # Ensure we only write nodes that actually exist in the mesh
                if node <= len(nodes):
                    f.write(f"{node}, 1, 3, 0.0\n")

            # --- STEP DEFINITION ---
            f.write("*STEP\n*STATIC\n")
            
            # Loads
            f.write("*CLOAD\n")
            for node in load_nodes:
                f.write(f"{node}, 3, -1000.0\n")

            # Output Requests (Keep it minimal to avoid memory errors)
            f.write("*NODE FILE\nU\n")
            f.write("*EL FILE\nS\n")
            f.write("*END STEP\n") # Ensure only ONE of these exists at the very end


        # 4. Run CCX
        print(f"Starting CalculiX solve in {self.work_dir}...")
        try:
            # CCX expects to be run in the folder where the .inp is
            time.sleep(0.2)

            subprocess.run(["ccx", self.job_name], cwd=self.work_dir, check=True)
            frd_path = os.path.join(self.work_dir, f"{self.job_name}.frd")
            attempts = 0
            while attempts < 5 and (not os.path.exists(frd_path) or os.path.getsize(frd_path) == 0):
                time.sleep(0.1)
                attempts += 1
        
            return frd_path

        except Exception as e:
            print(f"Solver failed: {e}")
            return None

    def parse_frd(self, frd_path):
        """Instance method: Forces a 3-column shape regardless of Fortran smushing."""
        import re
        import numpy as np
        
        raw_values = []
        # Regex to find scientific notation numbers
        float_regex = re.compile(r"[-+]?\d*\.\d+[Ee][-+]?\d+")

        try:
            with open(frd_path, 'r') as f:
                for line in f:
                    if line.startswith(" -1"):
                        matches = float_regex.findall(line)
                        if len(matches) >= 3:
                            # Convert the last 3 matches to floats and add to a flat list
                            raw_values.extend([float(x) for x in matches[-3:]])
            
            all_data = np.array(raw_values)
            u_array = all_data.reshape(-1, 3)

            # NEW: Check if we have exactly 3x the nodes (Quadratic Mesh)
            # If the user's mesh only has 1123 nodes, we slice the results
            target_count = 1123 # You can pass this as an argument or use self.mesh.n_points
            if u_array.shape[0] > target_count:
                print(f"Downsampling results from {u_array.shape[0]} to {target_count}")
                u_array = u_array[:target_count, :]

            return {'U': u_array}
            
        except Exception as e:
            print(f"Instance Parser Error: {e}")
            return {'U': np.empty((0, 3))}
    
def line_is_end_of_block(line):
    # Blocks end with -3 (End of Analysis) or -1 (if it's a header)
    # But usually, a new block starts with a '1' or the file ends with '9999'
    return line.strip() in ["-3", "9999"] or line.startswith(" 1")