from inspect_cad import read_parts, ROOT
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
import sys

parts = read_parts(ROOT / "assets/cad" / sys.argv[1])
for i in map(int, sys.argv[2:]):
    print("PART", i, parts[i][0])
    exp = TopExp_Explorer(parts[i][1], TopAbs_FACE)
    seen = set()
    while exp.More():
        s = BRepAdaptor_Surface(TopoDS.Face_s(exp.Current()))
        if s.GetType() == GeomAbs_Cylinder:
            c = s.Cylinder()
            p, d = c.Location(), c.Axis().Direction()
            row = tuple(round(x,3) for x in (c.Radius(), p.X(),p.Y(),p.Z(),d.X(),d.Y(),d.Z()))
            if row not in seen:
                seen.add(row)
                print(row)
        exp.Next()
