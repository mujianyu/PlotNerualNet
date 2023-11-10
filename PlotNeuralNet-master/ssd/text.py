import sys
import os
import subprocess
sys.path.append('../')
from pycore.tikzeng import *

'''
VGG 部分的网络结构 卷积核大小为3x3 步长为1 padding为1 pool层为2x2步长为2
300, 300, 3 
-> 300, 300, 64 -> 300, 300, 64 -> 150, 150, 64  CCM CONV1
-> 150, 150, 128 -> 150, 150, 128 -> 75, 75, 128 CCM CONV2

->75, 75, 256 -> 75, 75, 256 -> 75, 75, 256 -> 38, 38, 256 CCCM CONV3(ceilmode=True) 因为75除以2不是整数,所以ceilmode=True
-> 38, 38, 512 -> 38, 38, 512 -> 38, 38, 512(Conv4_3) -> 19, 19, 512  CCCM CONV4
->19, 19, 512 -> 19, 19, 512 -> 19, 19, 512  -> 19, 19, 512 CCCM CONV5 这里的pool5是3x3步长为1且padding为1
-> 19, 19, 1024 (fc6转换为conv6)
-> 19, 19, 1024 (fc7转换为conv7)

base = [64, 64, 'M', 
128, 128, 'M', 
256, 256, 256, 'C', 
512, 512, 512, 'M',
512, 512, 512]
'''
'''
添加了额外的层(前一个卷积核kernal_size=1 步长为1后一个卷积核步长为2 kernel_size=3 padding=1)
-> 19,19,256 -> 10,10,512 Conv8 CC
-> 10,10,128 -> 5,5,256 Conv9 CC
后面的卷积padding = 0 步长为1
-> 5,5,128 -> 3,3,256 Conv10 CC
-> 3,3,128 -> 1,1,256 Conv11 CC
'''

# 定义神经网络架构
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    to_input("street.jpg", to='(-1,0,0)', width=2, height=2, name="temp"),
    # s_filer表示该层的图像大小 (需要自己计算),n_filer表示输入通道和输出通道大小 (自己设定)
    # offset表示该层的位置, to表示该层的输入来源, height表示该层的高度 h, depth表示该层的深度 w, width表示该层的宽度 c, caption表示该层的名称
    #s_filer=(300,300)表示该层的图像大小 (需要自己计算),n_filer=3表示输入通道和输出通道大小 (自己设定)
    #Conv1
    to_Conv("conv1_1", s_filer=(300,300), n_filer=64, offset="(0,0,0)", to="(0,0,0)", height=9, depth=9, width=1, caption='Conv1'),
    to_Conv("conv1_2", s_filer=(300,300), n_filer=64, offset="(0,0,0)", to="(conv1_1-east)", height=9, depth=9, width=1, caption=''),
    to_Pool("pool1",s_filer=(150,150), n_filer=64, offset="(0,0,0)", to="(conv1_2-east)", height=8, depth=8, width=1, caption=""),
    #Conv2
    to_Conv("conv2_1", s_filer=(150,150), n_filer=128, offset="(1,0,0)", to="(pool1-east)", height=8, depth=8, width=2, caption='Conv2'),
    to_connection("pool1", "conv2_1"),
    to_Conv("conv2_2", s_filer=(150,150), n_filer=128, offset="(0,0,0)", to="(conv2_1-east)", height=8, depth=8, width=2, caption=''),
    to_Pool("pool2",s_filer=(75,75), n_filer=128, offset="(0,0,0)", to="(conv2_2-east)", height=7, depth=7, width=2, caption=""),
    #Conv3
    to_Conv("conv3_1", s_filer=(75,75), n_filer=256, offset="(1,0,0)", to="(pool2-east)", height=7, depth=7, width=4, caption='Conv3'),
    to_connection("pool2", "conv3_1"),
    to_Conv("conv3_2", s_filer=(75,75), n_filer=256, offset="(0,0,0)", to="(conv3_1-east)", height=7, depth=7, width=4, caption=''),
    to_Conv("conv3_3", s_filer=(75,75), n_filer=256, offset="(0,0,0)", to="(conv3_2-east)", height=7, depth=7, width=4, caption=''),
    to_Pool("pool3",s_filer=(38,38), n_filer=256, offset="(0,0,0)", to="(conv3_3-east)", height=6, depth=6, width=4, caption=""),
    #Conv4
    to_Conv("conv4_1", s_filer=(38,38), n_filer=512, offset="(1,0,0)", to="(pool3-east)", height=6, depth=6, width=8, caption='Conv4'),
    to_connection("pool3", "conv4_1"),
    to_Conv("conv4_2", s_filer=(38,38), n_filer=512, offset="(0,0,0)", to="(conv4_1-east)", height=6, depth=6, width=8, caption=''),
    to_Conv("conv4_3", s_filer=(38,38), n_filer=512, offset="(0,0,0)", to="(conv4_2-east)", height=6, depth=6, width=8, caption=''),
    to_Pool("pool4",s_filer=(19,19), n_filer=512, offset="(0,0,0)", to="(conv4_3-east)", height=5, depth=5, width=8, caption=""),
    #Conv5
    to_Conv("conv5_1", s_filer=(19,19), n_filer=512, offset="(1,0,0)", to="(pool4-east)", height=5, depth=5, width=8, caption='Conv5'),
    to_connection("pool4", "conv5_1"),
    to_Conv("conv5_2", s_filer=(19,19), n_filer=512, offset="(0,0,0)", to="(conv5_1-east)", height=5, depth=5, width=8, caption=''),
    to_Conv("conv5_3", s_filer=(19,19), n_filer=512, offset="(0,0,0)", to="(conv5_2-east)", height=5, depth=5, width=8, caption=''),
    to_Pool("pool5",s_filer=(19,19), n_filer=512, offset="(0,0,0)", to="(conv5_3-east)", height=5, depth=5, width=8, caption=""),
    #FC6 19, 19, 1024 (fc6转换为conv6)
    to_Conv("conv6", s_filer=(19,19), n_filer=1024, offset="(1,0,0)", to="(pool5-east)", height=5, depth=5, width=16, caption='Conv6'),
    to_connection("pool5", "conv6"),
    #FC7 19, 19, 1024 (fc7转换为conv7)
    to_Conv("conv7", s_filer=(19,19), n_filer=1024, offset="(1,0,0)", to="(conv6-east)", height=5, depth=5, width=16, caption='Conv7'),
    to_connection("conv6", "conv7"),


    #Conv8
    to_Conv("conv8_1", s_filer=(19,19), n_filer=256, offset="(1,0,0)", to="(conv7-east)", height=5, depth=5, width=4, caption='Conv8'),
    to_connection("conv7", "conv8_1"),
    to_Conv("conv8_2", s_filer=(10,10), n_filer=512, offset="(0,0,0)", to="(conv8_1-east)", height=4, depth=4, width=8, caption=''),

    #Conv9
    to_Conv("conv9_1", s_filer=(10,10), n_filer=128, offset="(1,0,0)", to="(conv8_2-east)", height=4, depth=4, width=2, caption='Conv9'),
    to_connection("conv8_2", "conv9_1"),
    to_Conv("conv9_2", s_filer=(5,5), n_filer=256, offset="(0,0,0)", to="(conv9_1-east)", height=3, depth=3, width=4, caption=''),
    #Conv10
    to_Conv("conv10_1", s_filer=(5,5), n_filer=128, offset="(1,0,0)", to="(conv9_2-east)", height=3, depth=3, width=2, caption='Conv10'),
    to_connection("conv9_2", "conv10_1"),
    to_Conv("conv10_2", s_filer=(3,3), n_filer=256, offset="(0,0,0)", to="(conv10_1-east)", height=2, depth=2, width=4, caption=''),
    #Conv11
    to_Conv("conv11_1", s_filer=(3,3), n_filer=128, offset="(1,0,0)", to="(conv10_2-east)", height=2, depth=2, width=2, caption='Conv11'),
    to_connection("conv10_2", "conv11_1"),
    to_Conv("conv11_2", s_filer=(1,1), n_filer=256, offset="(0,0,0)", to="(conv11_1-east)", height=1, depth=1, width=4, caption=''),
    to_end()                           
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

    # 使用 LaTeX 编译器将 .tex 文件转换为 .pdf 文件
    subprocess.call([r'D:\MiKTeX\miktex\bin\x64\pdflatex.exe', namefile + '.tex'])

    pdf_file = namefile + '.pdf'
    image_file = namefile + '.png'

    subprocess.call([r'D:\gs\bin\gswin64c.exe', '-sDEVICE=pngalpha', '-o', image_file, '-r300', pdf_file])

    # 删除中间生成的文件
    cleanup(namefile)


def cleanup(namefile):
    # 删除中间生成的文件
    extensions = ['.aux', '.log', '.tex']
    for ext in extensions:
        filename = namefile + ext
        if os.path.exists(filename):
            os.remove(filename)


if __name__ == '__main__':
    main()

