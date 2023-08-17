import copy
import math
import numpy as np
import torch
import json 
class WireframeGraph:
    def __init__(self, 
                vertices: torch.Tensor, 
                v_confidences: torch.Tensor,
                edges: torch.Tensor, 
                edge_weights: torch.Tensor, 
                frame_width: int, 
                frame_height: int):
        self.vertices = vertices.clone().detach()
        self.v_confidences = v_confidences.clone().detach()
        self.edges = edges.clone().detach()
        self.weights = edge_weights.clone().detach()
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def line_segments(self, threshold = 0.97):
        is_valid = self.weights>threshold
        p1 = self.vertices[self.edges[is_valid,0]]
        p2 = self.vertices[self.edges[is_valid,1]]
        ps = self.weights[is_valid]

        lines = torch.cat((p1,p2,ps[:,None]),dim=-1)

        return lines

        
    def rescale(self, image_width, image_height):
        scale_x = float(image_width)/float(self.frame_width)
        scale_y = float(image_height)/float(self.frame_height)

        self.vertices[:,0] *= scale_x
        self.vertices[:,1] *= scale_y
        self.frame_width = image_width
        self.frame_height = image_height

    def jsonize(self):
        return {
            'vertices': self.vertices.cpu().tolist(),
            'vertices-score': self.v_confidences.cpu().tolist(),
            'edges': self.edges.cpu().tolist(),
            'edges-weights': self.weights.cpu().tolist(),
            'height': self.frame_height,
            'width': self.frame_width,
        }

    @classmethod
    def load_json(cls,path):
        with open(path,'r') as f:
            data = json.load(f)
        data['vertices'] = torch.tensor(data['vertices'],dtype=torch.float32)
        data['vertices-score'] = torch.tensor(data['vertices-score'],dtype=torch.float32)
        data['edges'] = torch.tensor(data['edges'],dtype=torch.long)
        data['edges-weights'] = torch.tensor(data['edges-weights'])
        out = cls(
            torch.tensor(data['vertices'],dtype=torch.float32,requires_grad=False),
            torch.tensor(data['vertices-score'],dtype=torch.float32,requires_grad=False),
            torch.tensor(data['edges'],dtype=torch.long,requires_grad=False),
            torch.tensor(data['edges-weights'],dtype=torch.float32,requires_grad=False),
            frame_width = data['width'],
            frame_height = data['height']
        )
        
        return out

    def __repr__(self) -> str:
        return "WireframeGraph\n"+\
               "Vertices: {}\n".format(self.vertices.shape[0])+\
               "Edges: {}\n".format(self.edges.shape[0],) + \
               "Frame size (HxW): {}x{}".format(self.frame_height,self.frame_width)
