import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt',type=str,required=True)

    opt = parser.parse_args()

    with open(opt.txt,'r') as f:
        data = f.readlines()

    lines3d = []

    for d in data:
        dv = d.split()
        line = [float(d) for d in dv[1:7]]
        lines3d.append(line)
    lines3d = np.array(lines3d)
    lines3d = lines3d.reshape(-1,2,3)

    np.savez(opt.txt[:-3]+'npz',lines3d=lines3d)

if __name__ == "__main__":
    main()