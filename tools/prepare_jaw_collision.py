"""Export the complete, un-decimated rigid jaws from the cached STEP tessellation."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=np.load(ROOT/'assets/SO101_Assembly_raw.npz')
    report={'source':'SO101 Assembly.step','method':'full STEP tessellation; coincident vertices welded; no decimation or convex hull','parts':{}}
    for name,index,link in [('fixed_jaw',8,5),('moving_jaw',9,6)]:
        original=source[f'v{index}']*.001
        vertices,inverse=np.unique(np.round(original,10),axis=0,return_inverse=True)
        faces=inverse[source[f'f{index}']]
        triangles=vertices[faces]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
        faces=faces[area>1e-15]
        path=ROOT/'assets/meshes'/f'{name}_collision.obj'
        with path.open('w') as stream:
            stream.write('# Full STEP jaw tessellation, meters in assembly reference frame; no decimation\n')
            for v in vertices:stream.write('v %.10f %.10f %.10f\n'%tuple(v))
            for f in faces+1:stream.write('f %d %d %d\n'%tuple(f))
        tri=vertices[faces];volumes=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6
        center=(tri.sum(1)*volumes[:,None]).sum(0)/(4*volumes.sum())
        report['parts'][name]=dict(file=path.name,link=link,source_part_index=index,
            source_triangles=len(source[f'f{index}']),triangles=len(faces),vertices=len(vertices),
            solid_volume_m3=abs(float(volumes.sum())),center_of_mass_ref_m=center.tolist(),
            dimensions_m=np.ptp(vertices,axis=0).tolist())
    (ROOT/'assets/jaw_collision.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
