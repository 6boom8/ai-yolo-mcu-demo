
  ✨ 效果展示 - YOLOv8 摄像头实时目标检测
   支持调用本地摄像头进行视频流抓拍，实时输出目标检测结果。
<img width="1706" height="1279" alt="702b9043d10282b153ae21d43d2a676f" src="https://github.com/user-attachments/assets/daa0756f-3715-4ecd-bbd2-80f8166aa863" />

# ai-yolo-mcu-demo
Local LLM+YOLO object detection+Arduino control
 import streamlit as st
from ultralytics import YOLO
from PIL import Image  # 新增这一行，导入图片处理库

st.title("我的第一个AI识图网页 🐶🐱")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("请上传一张图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. 把上传的文件打开，转成 PIL 图片
    image = Image.open(uploaded_file)
    
    # 显示原图
    st.image(image, caption="你上传的原图", use_container_width=True)
    
    with st.spinner("AI正在看图，请稍候..."):
        # 2. 把 PIL 图片喂给模型（不再用 uploaded_file）
        results = model(image)
        
        # 3. 获取画好框框的图片
        # 注意：YOLO输出的图片颜色是BGR格式（适合OpenCV），网页需要RGB格式
        result_image = results[0].plot()
        result_image_rgb = result_image[:, :, ::-1] # 这行代码把 BGR 变成 RGB，防止照片颜色变蓝
        
        # 显示AI处理后的图
        st.image(result_image_rgb, caption="AI识别结果", use_container_width=True)
<img width="1279" height="1706" alt="7216956acd84defae753fd7c82304ba1" src="https://github.com/user-attachments/assets/9d2d0a15-706a-4b74-a10a-bd59a0d16d0f" />
<img width="1706" height="1279" alt="019d1818dec7500ec627e3025ee1c5fd" src="https://github.com/user-attachments/assets/93e3209b-7cf2-4bc0-8e4b-cf4b783afd47" />
基于 ultralytics 库，使用 coco128 数据集对 yolov8n.pt 进行了 5 个 epoch 的微调训练。训练日志已截图，验证集指标 mAP 达 0.58。训练权重已通过 .gitignore 忽略，不在仓库中上传。
<img width="1706" height="1279" alt="36b03525146c776cd814aab39f00bda5" src="https://github.com/user-attachments/assets/537401ea-661a-4e53-b820-f4c2b091dc02" />


