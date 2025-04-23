import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import sys

# اضافه کردن مسیر DeOldify به sys.path
sys.path.append(r"H:\tutorials\Python tutorial\Projects\DeOldify")

# وارد کردن ماژول‌های DeOldify
try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    from deoldify.visualize import get_image_colorizer
except ImportError as e:
    print(f"خطا در وارد کردن DeOldify: {e}")
    print("لطفاً مطمئن شوید که DeOldify نصب شده است: pip install deoldify")
    sys.exit(1)

sys.path.append(r"H:\tutorials\Python tutorial\Projects\GFPGAN")
from gfpgan import GFPGANer

# تنظیم دستگاه (GPU)
device.set(device=DeviceId.GPU0)

# مسیر پوشه‌های ورودی و خروجی
input_folder = "input_photos"
output_folder = "output_photos"
Path(output_folder).mkdir(parents=True, exist_ok=True)

# بارگذاری مدل DeOldify
colorizer = get_image_colorizer(
    render_factor=35,  # کاهش به 35 برای کاهش مصرف حافظه
    root_folder=Path(r"H:\tutorials\Python tutorial\Projects\DeOldify")
)

# غیرفعال کردن واترمارک
colorizer._add_default_watermark = False

# بارگذاری مدل GFPGAN
gfpgan = GFPGANer(
    model_path=r"H:\tutorials\Python tutorial\Projects\GFPGAN\experiments\pretrained_models\GFPGANv1.3.pth",
    upscale=1,  # کاهش به 1 برای کاهش مصرف حافظه
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)

# تابع برای پردازش یک عکس
def process_image(image_path, output_path):
    try:
        # خواندن عکس
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("عکس خوانده نشد")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # مرحله ۱: رنگی‌سازی با DeOldify
        colorized_img = colorizer.get_transformed_image(
            path=image_path,
            render_factor=35,
            post_process=True
        )
        colorized_img = np.array(colorized_img)

        # مرحله ۲: ارتقای کیفیت با GFPGAN
        _, _, enhanced_img = gfpgan.enhance(
            colorized_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        # ذخیره عکس نهایی
        enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, enhanced_img_bgr)
        print(f"پردازش {image_path} با موفقیت انجام شد!")

        # خالی کردن حافظه GPU
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"خطا در پردازش {image_path}: {e}")

# پردازش تمام عکس‌های داخل پوشه
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        process_image(image_path, output_path)

print("پردازش همه عکس‌ها کامل شد!")