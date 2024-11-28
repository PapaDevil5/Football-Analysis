from ultralytics import YOLO

model=YOLO('models/best.pt')
result=model.predict('videos/demo_vid_2',save=True)
print(result[0])
print('=====================================================')
for box in result[0].boxes:
    print(box)

