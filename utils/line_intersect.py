from tkinter import E
from IPython import embed
import torch
# Source: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
  
# Given three collinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r):  
    p_ = p.unsqueeze(1)
    q_ = q.unsqueeze(0)
    r_ = r.unsqueeze(1)
 
    a, _ = torch.max(torch.cat([p_[:, :, 0], r_[:, :, 0]], dim=1), dim=1, keepdim=True)
    b = q_[:, :, 0] <= a

    c, _ = torch.min(torch.cat([p_[:, :, 0], r_[:, :, 0]], dim=1), dim=1, keepdim=True)
    d = q_[:, :, 0] >= c

    e, _ = torch.max(torch.cat([p_[:, :, 1], r_[:, :, 1]], dim=1), dim=1, keepdim=True)
    f = q_[:, :, 1] <= e

    g, _ = torch.min(torch.cat([p_[:, :, 1], r_[:, :, 1]], dim=1), dim=1, keepdim=True)
    h = q_[:, :, 1] >= g

    bd = torch.logical_and(b, d)
    fh = torch.logical_and(f, h)

    i = torch.logical_and(bd, fh)
    return i

    # if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
    #        (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
    #     return True
    # return False
  
def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Collinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  

    p_ = p.unsqueeze(1)
    q_ = q.unsqueeze(1)
    r_ = r.unsqueeze(0)
 
    a = (q_[:, :, 1] - p_[:, :, 1])
    b = (r_[:, :, 0] - q_[:, :, 0]) 

    c = (q_[:, :, 0]-p_[:, :, 0]) 
    d = (r_[:, :, 1]-q_[:, :, 1]) 

    val = a * b - c * d
    orientation = torch.zeros_like(val)
    orientation[val>0] = 1
    orientation[val<0] = 2
    return orientation


    # val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    # if (val > 0):  
    #     # Clockwise orientation 
    #     return 1
    # elif (val < 0):  
    #     # Counterclockwise orientation 
    #     return 2
    # else:  
    #     # Collinear orientation 
    #     return 0
  
# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    o3 = o3.transpose(1, 0)
    o4 = o4.transpose(1, 0)

    # # General case 
    result = torch.zeros_like(o1)
    result[torch.logical_and(o1 != o2, o3 != o4)] = 1
    
  
    # Special Cases 
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
    result[torch.logical_and(o1 == 0, onSegment(p1, p2, q1))] = 1

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    result[torch.logical_and(o2 == 0, onSegment(p1, q2, q1))] = 1

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    result[torch.logical_and(o3 == 0, onSegment(p2, p1, q2).transpose(1, 0))] = 1

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
    result[torch.logical_and(o4 == 0, onSegment(p2, q1, q2).transpose(1, 0))] = 1
    
    result = (result==1)
    return result 
    # # General case 
    # if ((o1 != o2) and (o3 != o4)): 
    #     return True
  
    # # Special Cases 
  
    # # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
    # if ((o1 == 0) and onSegment(p1, p2, q1)): 
    #     return True
  
    # # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
    # if ((o2 == 0) and onSegment(p1, q2, q1)): 
    #     return True
  
    # # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
    # if ((o3 == 0) and onSegment(p2, p1, q2)): 
    #     return True
  
    # # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
    # if ((o4 == 0) and onSegment(p2, q1, q2)): 
    #     return True
  
    # # If none of the cases 
    # return False


