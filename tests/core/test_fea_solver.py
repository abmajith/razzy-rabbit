import pytest
import meshio
import os
import shutil
import numpy as np
from razzy_rabbit.core.fea_solver import FEASolver

def test_calculix_full_pipeline(tmp_path):
    """Verifies: Write .inp -> Run CCX -> Parse .frd with Meshio."""
    work_dir = tmp_path / "fea_test"
    work_dir.mkdir()
    
    # Ensure ccx is actually executable
    ccx_bin = shutil.which("ccx")
    assert ccx_bin is not None, "ccx not found in PATH"

    # 2. Create a minimal .frd-compatible input (A single tet)
    # Usually, we'd use solver.write_inp(), but for a unit test, 
    # let's write a known-good minimal block.
    inp_content = """*NODE
1, 0, 0, 0
2, 1, 0, 0
3, 0, 1, 0
4, 0, 0, 1
*ELEMENT, TYPE=C3D4, ELSET=E1
1, 1, 2, 3, 4
*MATERIAL, NAME=STEEL
*ELASTIC
210000, 0.3
*SOLID SECTION, ELSET=E1, MATERIAL=STEEL
*BOUNDARY
** Fix Node 1 completely (no translation)
1, 1, 3, 0
** Fix Node 2 in Y and Z
2, 2, 3, 0
** Fix Node 3 in Z
3, 3, 3, 0
*STEP
*STATIC
*CLOAD
4, 3, -100
*NODE FILE
U
*EL FILE
S
*END STEP
"""
    inp_path = work_dir / "razzy_job.inp"
    inp_path.write_text(inp_content)

    # 3. Execute solver
    import subprocess
    # Execute solver with error capturing
    result = subprocess.run(
        [ccx_bin, "razzy_job"], 
        cwd=str(work_dir), 
        capture_output=True, 
        text=True
    )
    
    # If CCX failed, this will show you the EXACT Fortran/CalculiX error
    assert result.returncode == 0, f"CCX Failed! Output: {result.stdout}\nError: {result.stderr}"
    
    frd_path = work_dir / "razzy_job.frd"
    assert frd_path.exists(), "CalculiX finished but razzy_job.frd is missing"


    # 4. Manual Verification (Since meshio dropped .frd support)
    with open(frd_path, 'r') as f:
        content = f.read()
    
    # Check for the nodal displacement block header in FRD
    # ' -1' indicates a nodal results block
    assert " -1" in content, "Displacement data block missing from .frd"
    
    # Check for our 4 nodes
    lines = content.splitlines()
    data_lines = [l for l in lines if l.startswith(" -1")]
    assert len(data_lines) >= 4, f"Expected 4 result nodes, found {len(data_lines)}"
    
    print("Solver Success: FRD contains valid nodal data.")

    solver = FEASolver(None)
    results = solver.parse_frd(frd_path)
    # Ensure it's N rows by 3 columns
    assert results['U'].ndim == 2
    assert results['U'].shape[1] == 3