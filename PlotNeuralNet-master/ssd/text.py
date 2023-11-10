import sys
import os
import subprocess
sys.path.append('../')
from pycore.tikzeng import *


# 定义神经网络架构
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    to_input("street.jpg", to='(-5,0,0)', width=6, height=6, name="temp"),
    # s_filer表示该层的图像大小 (需要自己计算),n_filer表示输入通道和输出通道大小 (自己设定)
    # offset表示该层的位置, to表示该层的输入来源, height表示该层的高度 h, depth表示该层的深度 w, width表示该层的宽度 c, caption表示该层的名称
    #s_filer=(256,256)表示该层的图像大小 (需要自己计算),n_filer=3表示输入通道和输出通道大小 (自己设定)
    to_Conv("conv1", s_filer=(256,256), n_filer=3, offset="(0,0,0)", to="(0,0,0)", height=50, depth=50, width=3, caption='CONV1'),
    to_Pool("pool1", s_filer=(256,256),n_filer =3,offset="(0,0,0)", to="(conv1-east)", height=32, depth=32, width=3, caption="MaxPool1"),


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

