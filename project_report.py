import os
import subprocess

print("\n===== PROJECT REPORT =====\n")

# 1. Python & environment info
print("🔹 Python Version:")
subprocess.run(["python", "--version"])

print("\n🔹 Python Path:")
subprocess.run(["which", "python"])


# 2. Installed packages
print("\n===== INSTALLED PACKAGES =====")
subprocess.run(["pip", "list"])


# 3. Project structure
print("\n===== PROJECT STRUCTURE =====")

for root, dirs, files in os.walk(".", topdown=True):
    level = root.replace(os.getcwd(), "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")


# 4. Check important folders/files
print("\n===== CHECKS =====")

def check_path(path):
    if os.path.exists(path):
        print(f"✅ Found: {path}")
    else:
        print(f"❌ Missing: {path}")

# dataset
check_path("Dataset/HAM10000_metadata.csv")
check_path("Dataset/HAM10000_images_part_1")
check_path("Dataset/HAM10000_images_part_2")

# model
check_path("model/skin_model.pth")

# core files
check_path("train.py")
check_path("app.py")
check_path("main.py")
check_path("utils.py")
check_path("skin_disease/main.py")


# 5. Sample dataset check
print("\n===== SAMPLE DATA CHECK =====")

image_count = 0
for folder in ["Dataset/HAM10000_images_part_1", "Dataset/HAM10000_images_part_2"]:
    if os.path.exists(folder):
        count = len(os.listdir(folder))
        print(f"{folder}: {count} files")
        image_count += count

print(f"\nTotal images found: {image_count}")

print("\n===== END OF REPORT =====\n")