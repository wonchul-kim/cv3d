import cv2
import os
from natsort import natsorted

# 이미지들이 있는 폴더 경로
image_folder = '/HDD/etc/outputs/calibration_single_tag/tracking'  # 예: 'C:/project/images'
output_video = '/HDD/etc/outputs/calibration_single_tag/tracking/output.mp4'
import imageio
fps = 2                       # 프레임 속도

# 이미지 파일 목록 불러오기
images = [img for img in os.listdir(image_folder) if img.lower().endswith('.png')]
images = natsorted(images)

if not images:
    raise ValueError("❌ PNG 이미지가 없습니다. 'images' 폴더를 확인하세요.")

# 첫 번째 이미지로 크기 확인
first_frame = imageio.imread(os.path.join(image_folder, images[0]))
height, width = first_frame.shape[:2]

# 비디오 저장
writer = imageio.get_writer(output_video, fps=fps, codec='libx264', format='FFMPEG')

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = imageio.imread(img_path)
    writer.append_data(frame)

writer.close()

print(f"✅ 동영상 생성 완료: {output_video}")
print("💡 이 파일은 Notion, 브라우저, 모바일에서도 바로 재생됩니다.")
