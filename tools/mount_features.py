from pathlib import Path
import numpy as np
import trimesh
from collections import defaultdict

root=Path(__file__).resolve().parents[1]
m=trimesh.load(root/'assets/cad/IMXmountSO101.stl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,8))
for edge in m.facets_boundary[1]:
    p=m.vertices[edge]
    ax.plot(p @ np.array([.42261826,.90630779,0]),p[:,2],color='black',lw=.7)
ax.set_aspect('equal'); ax.grid(); ax.set_xlabel('Along plate (mm)'); ax.set_ylabel('Z (mm)')
fig.savefig(root/'assets/mount_plate_inspection.png',dpi=140)
for k in np.argsort(m.facets_area)[-8:][::-1]:
    edges=m.facets_boundary[k]
    adj=defaultdict(list)
    for a,b in edges:
        adj[a].append(b); adj[b].append(a)
    unseen=set(adj)
    print('FACET', k, 'area',m.facets_area[k], 'normal',m.facets_normal[k])
    while unseen:
        stack=[next(iter(unseen))]; ids=set()
        while stack:
            i=stack.pop()
            if i in ids: continue
            ids.add(i); unseen.discard(i)
            stack.extend(j for j in adj[i] if j not in ids)
        p=m.vertices[list(ids)]
        print('loop',len(ids),'mean',p.mean(0),'extent',np.ptp(p,axis=0))
