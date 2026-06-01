# Data Directory

This directory contains data files used by the example notebooks.

## Files

### `moomoo.off`

A 3D triangular mesh in Object File Format (OFF).

**Format:** OFF (Object File Format)
- Simple ASCII format for 3D geometry
- Contains vertex coordinates and face indices
- Commonly used for storing polygonal meshes

**Usage:**
This mesh is used in the `examples/mesh.ipynb` notebook to demonstrate optimal transport on 3D mesh graphs.

**File structure:**
```
OFF
n_vertices n_faces n_edges
x1 y1 z1
x2 y2 z2
...
3 v1 v2 v3
3 v4 v5 v6
...
```

## Adding Your Own Data

To use your own mesh data:

1. Save your mesh in OFF format (or convert using tools like MeshLab)
2. Place the file in this directory
3. Modify the notebook to load your file:
   ```python
   vertices, faces = load_off_file('../data/your_mesh.off')
   ```

## Notes

- This directory is in `.gitignore` by default to avoid committing large data files
- Only essential example data files should be tracked in git
- For large datasets, consider hosting externally and providing download scripts
