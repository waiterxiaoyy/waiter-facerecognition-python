from PIL import Image

from facenet import Facenet

if __name__ == "__main__":
    model = Facenet()

    # image_1 = input('Input image_1 filename:')
    image_1 = "img/4_1.jpg"
    try:
        image_1 = Image.open(image_1)
    except:
        print('Image_1 Open Error! Try again!')

    # image_2 = input('Input image_2 filename:')
    image_2 = "img/4_2.jpg"
    try:
        image_2 = Image.open(image_2)
    except:
        print('Image_2 Open Error! Try again!')
    probability = model.detect_image(image_1,image_2)
    if(probability[0] < 0.9):
        print("Same Sample")
    else:
        print("Different Sample")
    print(probability[0])
    print(probability)
