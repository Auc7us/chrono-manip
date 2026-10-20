"""Read STEP assemblies with their placements intact (cadquery-ocp required)."""
from pathlib import Path
import json
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_LabelSequence, TDF_Label
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TopLoc import TopLoc_Location
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.BRep import BRep_Tool

ROOT = Path(__file__).resolve().parents[1]

def label_name(label):
    attr = TDataStd_Name()
    return attr.Get().ToExtString() if label.FindAttribute(TDataStd_Name.GetID_s(), attr) else "unnamed"

def read_parts(path):
    doc = TDocStd_Document(TCollection_ExtendedString("cad"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    if int(reader.ReadFile(str(path))) != 1 or not reader.Transfer(doc):
        raise RuntimeError(f"Cannot read {path}")
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    st.GetFreeShapes(roots)
    parts = []

    def visit(label, parent_loc, names):
        loc = parent_loc.Multiplied(st.GetLocation_s(label))
        ref = TDF_Label()
        definition = ref if st.GetReferredShape_s(label, ref) else label
        name = label_name(definition)
        names = names + [name]
        children = TDF_LabelSequence()
        if st.GetComponents_s(definition, children):
            for i in range(1, children.Length() + 1):
                visit(children.Value(i), loc, names)
        else:
            shape = st.GetShape_s(definition).Moved(loc)
            box = Bnd_Box()
            BRepBndLib.AddOptimal_s(shape, box, False, False)
            if not box.IsVoid():
                parts.append((names, shape, list(box.Get())))

    for i in range(1, roots.Length() + 1):
        visit(roots.Value(i), TopLoc_Location(), [])
    return parts

def tessellate(shape):
    BRepMesh_IncrementalMesh(shape, 0.16, False, 0.30, True)
    vertices, faces = [], []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            start = len(vertices)
            for j in range(1, tri.NbNodes() + 1):
                p = tri.Node(j).Transformed(loc.Transformation())
                vertices.append((p.X(), p.Y(), p.Z()))
            for j in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(j).Get()
                if face.Orientation() == TopAbs_REVERSED:
                    b, c = c, b
                faces.append((start + a - 1, start + b - 1, start + c - 1))
        exp.Next()
    return vertices, faces

if __name__ == "__main__":
    import sys
    path = ROOT / "assets/cad" / sys.argv[1]
    parts = read_parts(path)
    report = [{"index": i, "path": names, "bounds_mm": box} for i, (names, shape, box) in enumerate(parts)]
    target = ROOT / "assets" / (path.stem.replace(" ", "_") + "_parts.json")
    target.write_text(json.dumps(report, indent=2))
    print(target, len(report), "parts")
    for p in report:
        print(p["index"], "/".join(p["path"]), [round(v, 2) for v in p["bounds_mm"]])
    if "--mesh" in sys.argv:
        import numpy as np
        target = ROOT / "assets" / (path.stem.replace(" ", "_") + "_raw.npz")
        data = {}
        for i, (_, shape, _) in enumerate(parts):
            v, f = tessellate(shape)
            data[f"v{i}"] = np.array(v)
            data[f"f{i}"] = np.array(f)
        np.savez_compressed(target, **data)
        print("Saved", target)
