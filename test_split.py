import os
import shutil
import random

SOURCE_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\Data"
TEST_DIR = r"C:\Users\Tanya\Desktop\OASIS_dataset\Test"

TEST_SPLIT = 0.10   # 10% of data

os.makedirs(TEST_DIR, exist_ok=True)

classes = os.listdir(SOURCE_DIR)

for cls in classes:
    cls_source = os.path.join(SOURCE_DIR, cls)
    cls_target = os.path.join(TEST_DIR, cls)

    os.makedirs(cls_target, exist_ok=True)

    images = os.listdir(cls_source)
    total = len(images)
    test_count = int(total * TEST_SPLIT)

    print(f"{cls}: moving {test_count}/{total} images to TEST folder")

    test_images = random.sample(images, test_count)

    for img in test_images:
        src = os.path.join(cls_source, img)
        dst = os.path.join(cls_target, img)
        shutil.move(src, dst)

print("\n🎉 Test dataset created successfully!")
