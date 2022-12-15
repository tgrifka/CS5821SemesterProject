from matplotlib import pyplot as plt
import glob
from PIL import Image
import math
def main():
    images = glob.glob("C:\\Users\\grifk\\OneDrive\\Documents\\SchoolStuff\\CS5821\\SemesterProject\\Code\\Images\\*\\*.png")
    print(len(images))
    imgSizes = {}
    c = 0 #testing var
    for imgPath in images:
        try:
            with Image.open(imgPath) as img:
                size = img.size
                if size in imgSizes.keys():
                    imgSizes[size] += 1
                else:
                    imgSizes[size] = 1

        except Exception or OSError:
            print(imgPath)
        c += 1
        if c % 1000 == 0:
            print(f"Has Completed {c} images\n")
    print(imgSizes)

    keyStrings = []
    for k in imgSizes.keys():
        s = '('+str(k[0])+','+str(k[1])+')'
        keyStrings.append(s)

    f = plt.figure()
    f.set_figwidth(9.2)
    f.set_figheight(9.2)
    plt.bar(keyStrings, imgSizes.values())
    plt.xticks(rotation=90)
    plt.xlabel("Image Sizes")
    plt.ylabel("Number of Images")
    plt.title("Image Size Distribution")
    plt.show()

    minX = math.inf
    maxX = -math.inf
    minY = math.inf
    maxY = -math.inf

    totX = 0
    totY = 0
    aspect = 0
    for size in imgSizes:
        x = size[0]
        y = size[1]
        minX = min(minX, x)
        maxX = max(maxX, x)
        minY = min(minY, y)
        maxY = max(maxY, y)

        totX += (x * imgSizes[size])
        totY += (y * imgSizes[size])
        aspect += (round(x/y,2) * imgSizes[size])
        print(f"X: {x}, Y: {y}, Aspect Ratio: {round(x/y,2)}, Count:{imgSizes[size]}\n")


    print(f"Image Sizes are between: \n\tminX:{minX} \n\tmaxX:{maxX} \n\tminY:{minY} \n\tmaxY:{maxY}")
    avgX = totX/sum(imgSizes.values())
    avgY = totY/sum(imgSizes.values())
    avgAspect = aspect/sum(imgSizes.values())

    print(f"Average X:{avgX}")
    print(f"Average Y:{avgY}")
    print(f"Average Aspect: {avgAspect}")
if __name__ == '__main__':
    main()