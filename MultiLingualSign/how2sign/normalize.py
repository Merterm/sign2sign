from asyncore import read
import numpy as np 


# read dev.skeleton files.
def read_dev():
    data_path = "/Users/yangzhong/Downloads/dev.skels"
    
    
    neck_joints = [1e-8, 1e-8, 1e-8]
    left_shoulder_joints = []
    righ_shoulder_joints = []
    with open(data_path, mode='r', encoding='utf-8') as src_file:
        for line in src_file:
            trg_line = line.strip()
            trg_line = trg_line.split(" ")
            trg_line = [(float(joint) + 1e-8) for joint in trg_line]
            trg_size = 151
            skip_frames = 1
                # Split up the joints into frames, using trg_size as the amount of coordinates in each frame
                # If using skip frames, this just skips over every Nth frame
            # print(len(trg_line))
            trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]
            
            for frame in trg_frames:
                left_shoulder_joints.append(frame[6:9])
                righ_shoulder_joints.append(frame[15:18])
            print(left_shoulder_joints[-1])
            print(righ_shoulder_joints[-1])
            # # print(trg_frames[0] == trg_size * len(trg_frames))
            # print((trg_frames[0]))
            # # print(trg_frames[-1])
            # print()
            # print(trg_frames[10])
            # print()
            # print((trg_frames[-1]))
            # print()
            # print(len(trg_frames))
            
            # print(len(trg_frames) * 151)
            # # print(trg_frames[-1])
            # break
            
    avg_left_shoulder_joints = np.asarray(left_shoulder_joints).mean(axis=0)
    avg_righ_shoulder_joints = np.asarray(righ_shoulder_joints).mean(axis=0)
    print(avg_left_shoulder_joints)
    print(avg_righ_shoulder_joints)
if __name__ == "__main__":
    read_dev()