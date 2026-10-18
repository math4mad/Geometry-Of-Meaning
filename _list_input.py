import os
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count('/') - 2
    print("  " * depth + "/" + os.path.basename(root) + "/")
    for f in files[:5]:
        print("  " * (depth+1) + f)
    if depth > 3:
        break
print("=== model path search ===")
import glob
for p in glob.glob("/kaggle/input/**/config.json", recursive=True):
    print(p)
