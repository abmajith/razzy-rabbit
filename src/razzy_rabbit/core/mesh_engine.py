import gmsh
import tempfile
import os
import pyvista as pv
import numpy as np
import trimesh

def generate_tetra_mesh(pv_mesh, mesh_size=0.8):
    gmsh.initialize()
    gmsh.model.add("RobotPart")
    
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        stl_path = tmp.name
        pv_mesh.save(stl_path)
        gmsh.merge(stl_path)

    # --- ADVANCED HEALING ---
    # Force GMSH to re-classify the surfaces and ignore internal intersections
    angle = 40
    gmsh.model.mesh.classifySurfaces(angle * np.pi / 180., True, True)
    gmsh.model.mesh.createGeometry()
    
    # Create the volume loop
    surfaces = gmsh.model.getEntities(2)
    surf_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
    gmsh.model.geo.addVolume([surf_loop])
    gmsh.model.geo.synchronize()

    # Settings for CalculiX (C3D10)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # Avoids the meshio crash

    # Generate
    gmsh.model.mesh.generate(3)
    
    msh_path = os.path.join(tempfile.gettempdir(), "output.msh")
    gmsh.write(msh_path)
    gmsh.finalize()
    return pv.read(msh_path)


def generate_perfect_link():
    # Create two spheres (the ends) and a cylinder (the rod)
    s1 = trimesh.creation.uv_sphere(radius=1.5)
    s1.apply_translation([-5, 0, 0])
    
    s2 = trimesh.creation.uv_sphere(radius=1.5)
    s2.apply_translation([5, 0, 0])
    
    rod = trimesh.creation.cylinder(radius=0.8, height=10)
    rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    
    # BOOLEAN UNION: This is the "Magic" that makes it one solid
    # It removes all internal overlapping faces
    link = trimesh.boolean.union([s1, s2, rod])
    
    link.export("assets/link_rod.stl")
    print("Exported a water-tight link_rod.stl to assets/")