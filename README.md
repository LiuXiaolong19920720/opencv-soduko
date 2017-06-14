# opencv-soduko
---
#### This project aim to solve soduko in an image using opencv,which is implemented in Python.

- Get numbers in the image

- Recognise the numbers (Train a knn here)

- Generate soduko and solve it

- Display the output

---

<div>
<img src="https://github.com/LiuXiaolong19920720/opencv-soduko/blob/master/images/001.jpg" width="70%">
<br/>
<img src="https://github.com/LiuXiaolong19920720/opencv-soduko/blob/master/images/number-rec2.png" width="70%">
<br/>
<img src="https://github.com/LiuXiaolong19920720/opencv-soduko/blob/master/images/soduko.png" width="70%">
</div>

---

### How to use

`cd number`

run `mkdir.bat`

`cd ..`

`python preprocess.py`

`python knn.py`

---
### 细节参考文章
[OpenCV玩九宫格数独（零）——预告篇](https://mp.weixin.qq.com/s?__biz=MzU2MDAyNzk5MA==&mid=2247483693&idx=1&sn=c452c78d79bf9567dd33f70a84710d17&chksm=fc0f0114cb78880274a3ddd95ac0b7e3d05829245c14a05c241c5b2cd5fb03789b3af4343aef#rd)
[OpenCV玩九宫格数独（一）——九宫格图片中提取数字](https://mp.weixin.qq.com/s?__biz=MzU2MDAyNzk5MA==&mid=2247483695&idx=1&sn=e6068895b92749da4cf43c0ef7e87ed8&chksm=fc0f0116cb78880042081292735c468d1bbae57afb3e998468667d39a15a3bee7b1ebe7d9aa7#rd)
[OpenCV玩九宫格数独（二）：knn数字识别流程](https://mp.weixin.qq.com/s?__biz=MzU2MDAyNzk5MA==&mid=2247483701&idx=1&sn=72fd77773b876493d6e2a191dd781db7&chksm=fc0f010ccb78881a0ac4fb4a4ace52cbc03c980fbcc89e0cc1b1afd18274f23da17806a9206d#rd)
[OpenCV玩九宫格数独（三）：九宫格生成与数独求解](https://mp.weixin.qq.com/s?__biz=MzU2MDAyNzk5MA==&mid=2247483714&idx=1&sn=8ac4ea2da8f9b8bd75c7886b473d1104&chksm=fc0f017bcb78886de3ae2e89c71670cb91b6d776b8e5ebe8fe74876eb7b307eab694c801df29#rd)


