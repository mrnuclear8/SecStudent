#White-box-Cartoonization-master 使用说明
## 非原创声明
这个项目核心代码来源于 https://github.com/SystemErrorWang/White-box-Cartoonization

我们使用这个项目用作教师网络的实现，并采用源代码作者训练调优得到的能生成高质量图像的模型权重

## 环境安装


- 我们所使用的环境是conda虚拟环境，里面的pip包和conda已经全部打包到根目录，分别是**conda_env.yaml**和**pip_requirements.txt**安装命令如下:

    `conda env create -f conda_env.yaml`

    `pip install -r pip_requirements.txt`

- 原作者的 README.md 里也有环境安装介绍，若我们的方法不能成功，可以参考他的方案- 原作者的 README.md 里也有环境安装介绍，若我们的方法不能成功，可以参考他的方案
## 数据准备
- 采集得到数据应直接存放在 test_code/test_images 下，所有图片应直接放在该文件路径的根目录下
- 为了适应性能较差的训练硬件。我们对数据进行离线化处理。处理结果会缓存test_code/cartoonized_pic 运行前请确保硬盘中仍有与存入数据大小一致的空间

##  运行模型
- 运行 test_code/offline_prossessing.py 
  
- 运行 data_organization.py 

- 至此数据准备工作完成




