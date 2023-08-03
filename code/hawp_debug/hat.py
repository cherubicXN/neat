from sslib import WireframeGraph
from hawp import _C
from pathlib import Path
import glob
import torch
import numpy as np
import argparse
def wireframe2hafm(wireframe, dis_th=8, score_th=0.05, ang_th=0):

        lines = wireframe.line_segments(score_th)[:,:4]
        height = wireframe.frame_height
        width = wireframe.frame_width

        device = 'cuda'

        lines = lines.to(device=device)
        if lines.shape[0] == 0:
            hafm_ang = torch.zeros((3,height,width),device=device)
            hafm_dis = torch.zeros((1,height,width),device=device)
            hafm_mask = torch.zeros((1,height,width),device=device)
            return torch.zeros((3,height,width),device=device), torch.zeros((1,height,width),device=device), torch.zeros((1,height,width),device=device)
        lmap, _, _ = _C.encodels(lines,height,width,height,width,lines.size(0))
        dismap = torch.sqrt(lmap[0]**2+lmap[1]**2)[None]
        def _normalize(inp):
            mag = torch.sqrt(inp[0]*inp[0]+inp[1]*inp[1])
            return inp/(mag+1e-6)
        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])
        st_map = lmap[2:4]
        ed_map = lmap[4:]

        md_ = md_map.reshape(2,-1).t()
        st_ = st_map.reshape(2,-1).t()
        ed_ = ed_map.reshape(2,-1).t()
        Rt = torch.cat(
                (torch.cat((md_[:,None,None,0],md_[:,None,None,1]),dim=2),
                 torch.cat((-md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        R = torch.cat(
                (torch.cat((md_[:,None,None,0], -md_[:,None,None,1]),dim=2),
                 torch.cat((md_[:,None,None,1], md_[:,None,None,0]),dim=2)),dim=1)
        #Rtst_ = torch.matmul(Rt, st_[:,:,None]).squeeze(-1).t()
        #Rted_ = torch.matmul(Rt, ed_[:,:,None]).squeeze(-1).t()
        Rtst_ = torch.bmm(Rt, st_[:,:,None]).squeeze(-1).t()
        Rted_ = torch.bmm(Rt, ed_[:,:,None]).squeeze(-1).t()
        swap_mask = (Rtst_[1]<0)*(Rted_[1]>0)
        pos_ = Rtst_.clone()
        neg_ = Rted_.clone()
        temp = pos_[:,swap_mask]
        pos_[:,swap_mask] = neg_[:,swap_mask]
        neg_[:,swap_mask] = temp

        pos_[0] = pos_[0].clamp(min=1e-9)
        pos_[1] = pos_[1].clamp(min=1e-9)
        neg_[0] = neg_[0].clamp(min=1e-9)
        neg_[1] = neg_[1].clamp(max=-1e-9)
        
        mask = (dismap.view(-1)<=dis_th).float()

        pos_map = pos_.reshape(-1,height,width)
        neg_map = neg_.reshape(-1,height,width)

        md_angle  = torch.atan2(md_map[1], md_map[0])
        pos_angle = torch.atan2(pos_map[1],pos_map[0])
        neg_angle = torch.atan2(neg_map[1],neg_map[0])

        mask *= (pos_angle.reshape(-1)>ang_th*np.pi/2.0)
        mask *= (neg_angle.reshape(-1)<-ang_th*np.pi/2.0)

        pos_angle_n = pos_angle/(np.pi/2)
        neg_angle_n = -neg_angle/(np.pi/2)
        md_angle_n  = md_angle/(np.pi*2) + 0.5
        mask    = mask.reshape(height,width)


        hafm_ang = torch.cat((md_angle_n[None],pos_angle_n[None],neg_angle_n[None],),dim=0)
        hafm_dis   = dismap.clamp(max=dis_th)/dis_th
        mask = mask[None]
        return hafm_ang, hafm_dis, mask

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dis-th',default=4.0, type=float, 
        help='the distance threshold for foreground pixel controlling in HAT fields representation'
    )
    parser.add_argument('--ang-th',default=0.0, type=float, 
        help='the angle threshold for foreground pixel controlling in HAT fields representation'
    )

    parser.add_argument('--score-th', default=0.05, type=float,
        help='the score threshold for foreground pixel controlling in HAT fields representation'
    )

    parser.add_argument('--data', required=True, type=str, 
        help='the data'
    )

    parser.add_argument('--dest', required=True, type=str)
    args = parser.parse_args()

    return args


def main():
    import os.path as osp
    from tqdm import tqdm 

    args = parse_args()

    filenames = glob.glob(osp.join(args.data,'*.json'))

    DEST = Path(args.dest)
    DEST.mkdir(exist_ok=True)
    import pdb; pdb.set_trace()
    for f in tqdm(filenames):
        hafm_ang, hafm_dis, mask = wireframe2hafm(
            WireframeGraph.load_json(filenames[0]),
            score_th=0.05,
            dis_th=4,
        )

        hat = torch.cat((hafm_dis,hafm_ang),dim=0).permute((1,2,0)).contiguous().cpu().numpy()
        mask = mask[0].cpu().numpy()

        outpath = Path(f).with_suffix('.npz')
        outpath = DEST/outpath.name

        np.savez_compressed(outpath,hat=hat,mask=mask)

if __name__ == "__main__":
    main()