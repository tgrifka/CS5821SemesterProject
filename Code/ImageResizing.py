from matplotlib import pyplot as plt
import glob
from PIL import Image
import math
def main():
    images = glob.glob("C:\\Users\\grifk\\OneDrive\\Documents\\SchoolStuff\\CS5821\\SemesterProject\\Code\\Images\\*\\*.png")
    c=0
    for imgPath in images:
        try:
            with Image.open(imgPath) as img:
                newSizeImg = img.resize((616, 464))
                fp = imgPath
                fp = fp.replace('Images', 'ResizedImages')
                newSizeImg.save(fp)
        except Exception or OSError:
            print(fp)
        c += 1
        if c % 1000 == 0:
            print(f"Has Completed {c} images\n")

if __name__ == '__main__':
    main()