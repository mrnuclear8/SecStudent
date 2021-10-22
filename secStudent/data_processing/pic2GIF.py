# import matplotlib.pyplot as plt
import imageio,os
from PIL import Image
GIF=[]
filepath="C:/Users/XXX/PycharmProjects/White-box-Cartoonization-master/test_code/dataset_44G/trainA"#文件路径
filenames=os.listdir(filepath)

for i in range(0,10):
    print(filepath+"/"+filenames[i])
    GIF.append(imageio.imread(filepath+"/"+filenames[i]))

imageio.mimsave('C:/Users/XXX/PycharmProjects/White-box-Cartoonization-master/test_code/dataset_44G/result.gif',GIF,duration=0.1)#这个duration是播放速度，数值越小，速度越快
print('done')
