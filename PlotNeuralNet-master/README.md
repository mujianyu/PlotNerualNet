![text](https://gitee.com/mother-jianyu/PB/raw/master/images/text.png)

SSD输入图像大小（h,w,c）如上图变化

VGG 部分的网络结构 卷积核大小为3x3 步长为1 padding为1 pool层为2x2步长为2

300, 300, 3 

-> 300, 300, 64 -> 300, 300, 64 -> 150, 150, 64  CCM CONV1

-> 150, 150, 128 -> 150, 150, 128 -> 75, 75, 128 CCM CONV2

->75, 75, 256 -> 75, 75, 256 -> 75, 75, 256 -> 38, 38, 256 CCCM CONV3(ceilmode=True) 因为75除以2不是整数,所以ceilmode=True

-> 38, 38, 512 -> 38, 38, 512 -> 38, 38, 512(Conv4_3) -> 19, 19, 512  CCCM CONV4

->19, 19, 512 -> 19, 19, 512 -> 19, 19, 512  -> 19, 19, 512 CCCM CONV5 这里的pool5是3x3步长为1且padding为1

-> 19, 19, 1024 (fc6转换为conv6)

-> 19, 19, 1024 (fc7转换为conv7)

添加了额外的层(前一个卷积核kernal_size=1 步长为1后一个卷积核步长为2 kernel_size=3 padding=1)

-> 19,19,256 -> 10,10,512 Conv8 CC

-> 10,10,128 -> 5,5,256 Conv9 CC

后面的卷积padding = 0 步长为1

-> 5,5,128 -> 3,3,256 Conv10 CC

-> 3,3,128 -> 1,1,256 Conv11 CC



