from ctypes import *
import numpy as np
Sens = cdll.LoadLibrary('./Rasterizer/libSens.so')
from SensorData import SensorData
import os
import cv2

def LoadStreem(folder):

    _, _, files = next(os.walk(folder))
    frames = int(len(files)/4)

    cam2worlds = np.zeros((frames, 4, 4), dtype='float32')
    intrin =  np.loadtxt(f'{folder}/i_0000.txt')
    intrinsic = np.zeros((4,4), dtype='float32')
    intrinsic[0,0] = intrin[0]*(640/1920)
    intrinsic[1,1] = intrin[1]*(640/1920)
    intrinsic[0,2] = intrin[2]*(640/1920)
    intrinsic[1,2] = intrin[3]*(480/1440)
    intrinsic[2,2] = 1.0

    colors = np.zeros((frames, 480, 640, 3), dtype='uint8')
    depths = np.zeros((frames, 480, 640), dtype='float32')

    for index, i in enumerate(range(frames)):
        depth = cv2.imread(folder+"/frame-%06d.depth.png"%i, -1)
        depth = cv2.resize(depth, (640, 480))
        depth = depth/5.0 
        depths[i,:,:] = depth

        color = cv2.imread(folder+"/frame-%06d.color.png"%i)
        color = cv2.resize(color, (640, 480))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        colors[i,:,:,:] = color 
        
        pose = np.loadtxt(f'{folder}/p_{i:04}.txt')
        pose = pose.dot(
            np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        )
        #pose = np.linalg.inv(pose)
        cam2worlds[i,:,:] = pose 

    return colors, depths, cam2worlds, intrinsic 


def LoadSens(filename):
    sd = SensorData(filename)
    frames = len(sd.frames)
    color_height = sd.color_height
    color_width = sd.color_width
    depth_width = sd.depth_width
    depth_height = sd.depth_height

    cam2worlds = np.zeros((frames, 4, 4), dtype='float32')
    intrinsic =  sd.intrinsic_color #np.zeros((4,4), dtype='float32')
    colors = np.zeros((frames, color_height, color_width, 3), dtype='uint8')
    depths = np.zeros((frames, depth_height, depth_width), dtype='float32')

    for i in range(len(sd.frames)):
        depth_data = sd.frames[i].decompress_depth(sd.depth_compression_type)
        depth = np.fromstring(depth_data, dtype=np.uint16).reshape(depth_height, depth_width)
        depths[i,:,:] = depth

        color = sd.frames[i].decompress_color(sd.color_compression_type)
        colors[i,:,:,:] = color

        cam2worlds[i,:,:] = sd.frames[i].camera_to_world
    
    return colors, depths, cam2worlds, intrinsic


def LoadSens1(filename):
    print(filename)
    Sens.Parse(c_char_p(filename.encode('utf8')))
    depth_width = Sens.DW()
    depth_height = Sens.DH()
    color_width = Sens.CW()
    color_height = Sens.CH()
    frames = Sens.Frames()
    depths = np.zeros((frames, depth_height, depth_width), dtype='float32')
    colors = np.zeros((frames, color_height, color_width, 3), dtype='uint8')
    cam2worlds = np.zeros((frames, 4, 4), dtype='float32')
    intrinsic = np.zeros((4,4), dtype='float32')

    Sens.GetData(c_void_p(depths.ctypes.data), c_void_p(colors.ctypes.data),\
		c_void_p(cam2worlds.ctypes.data), c_void_p(intrinsic.ctypes.data))
    Sens.Clear()
    depths = np.nan_to_num(depths)
    return colors, depths, cam2worlds, intrinsic

def LoadOBJ(filename):
    lines = [l.strip() for l in open(filename)]
    V = []
    VT = []
    VN = []
    F = []
    FT = []
    FN = []
    for l in lines:
        words = [w for w in l.split(' ') if w != '']
        if len(words)==0: continue 
        if words[0] == 'v':
            V.append([float(words[1]), float(words[2]),
                     float(words[3])])
        elif words[0] == 'vt':
            VT.append([float(words[1]), float(words[2])])
        elif words[0] == 'vn':
            VN.append([float(words[1]), float(words[2]),
                      float(words[3])])
        elif words[0] == 'f':
            f = []
            ft = []
            fn = []
            for j in range(1, 4):
                w = words[j].split('/')
                f.append(int(w[0]) - 1)
                ft.append(int(w[1]) - 1)
                fn.append(int(w[2]) - 1)
            F.append(f)
            FT.append(ft)
            FN.append(fn)

    V = np.array(V, dtype='float32')
    VT = np.array(VT, dtype='float32')
    VN = np.array(VN, dtype='float32')
    F = np.array(F, dtype='int32')
    FT = np.array(FT, dtype='int32')
    FN = np.array(FN, dtype='int32')

    return (
        V,
        F,
        VT,
        FT,
        VN,
        FN,
        )


if __name__ == "__main__":
	import sys
	LoadSens(sys.argv[1])
	V,F,VT,FT,_,_ = LoadOBJ(sys.argv[2])
