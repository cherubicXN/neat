from collections import defaultdict
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colmap')

    opt = parser.parse_args()

    with open(opt.colmap,'r') as f:
        colmap = f.readlines()

    feature_tracks = {}    

    view_max = 0
    view_min = 100000

    feature_tracks_in_views = defaultdict(list)
    for line in colmap:
        if line[0] == '#':
            continue
        context = line.split()
        track_id = int(context[0])
        X = float(context[1])
        Y = float(context[2])
        Z = float(context[3])
        RGB = [int(c) for c in context[4:7]]
        ERR = context[7]

        view_ids = [int(c) for c in context[8::2]]
        feat_id = [int(c) for c in context[9::2]]
        feature_tracks[track_id] = {
            'x3d': np.array([X,Y,Z]),
            'views': np.array(view_ids),
            'featid': np.array(feat_id)
        }
        vmin = min(view_ids)
        vmax = max(view_ids)
        view_max = max(view_max,vmax)
        view_min = min(view_min,vmin)
        for v in view_ids:
            feature_tracks_in_views[v].append(track_id)
    
    v_cur = 1

    similarity = np.zeros(view_max-view_min+1)
    for v in range(view_min,view_max+1):
        if v == v_cur:
            continue
        ft_v = feature_tracks_in_views[v]
        ft_vc = feature_tracks_in_views[v_cur]
        ft_i = np.unique(np.array(ft_v+ft_vc))

        score = ft_i.shape[0]/(len(ft_v)+len(ft_vc))
        similarity[v-1] = score
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()