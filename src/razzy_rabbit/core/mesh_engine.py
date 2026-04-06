import gmsh
import tempfile
import os
import pyvista as pv

def generate_tetra_mesh(pv_mesh, mesh_size=1.5):
    """Scientific logic to convert Surface -> Volume."""
    gmsh.initialize()
    gmsh.model.add("CoreMesh")
    
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        pv_mesh.save(tmp.name)
        gmsh.merge(tmp.name)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(3)
    
    msh_path = os.path.join(tempfile.gettempdir(), "output.msh")
    gmsh.write(msh_path)
    gmsh.finalize()
    
    return pv.read(msh_path)
