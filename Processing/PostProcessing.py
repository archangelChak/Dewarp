import numpy as np
from IPython.display import Image
from PIL import Image, ImageDraw
import cv2
from IPython.display import display


def find_grid(smoothed_mask_in, mask_in):
    """
    Finds 11x11 points which divide document in 10x10 squares
    :param smoothed_mask_in: numpy.ndarray: smoothed mask without bias
    :param mask_in: numpy.ndarray: original predicted mask
    :return: numpy.ndarray: grid (11x11x2)
    """

    grid=[]
    for x in np.linspace(0, 1, 11):
        grid.append([])
        for y in np.linspace(0, 1, 11):
            idx = np.argmin((smoothed_mask_in[:, :, 0] - x)**2 + (smoothed_mask_in[:, :, 1] - y)**2)
            ind = np.unravel_index(idx, mask_in.shape[:2])
            grid[int(x*10)].append(ind)
    return grid


def get_image_from_boxes(image, box, height=75):
    """
    Cuts image with bounding box using perspective Transform
    :param image: numpy.ndarray: image
    :param box: numpy.ndarray: 2-D numpy array with coordinates
    :param height: int: height of the result image
    :return: numpy.ndarray: cut image
    """

    scale = np.sqrt((box[0][1] - box[1][1])**2 + (box[0][0] - box[1][0])**2) / height
    w = 65
    pts1 = np.float32(box)[:, ::-1]
    pts1 = pts1[[1, 0, 3, 2]]
    pts2 = np.float32([[0, 0], [height, 0], [0, w],  [height, w]])[:, ::-1]
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (w, height))
    return dst


def find_rect(grid_in, image_in):
    """
    Transforms image into list of cut rectangles from image using grid
    :param grid_in: numpy.ndarray: grid (11x11x2)
    :param image_in: numpy.ndarray: image
    :return: list: list of cut and transformed rectangles
    """

    boxes=[]
    rectangles=[]
    for i in range(10):
        for j in range(10):
            boxes.append([grid_in[i][j+1],grid_in[i][j],grid_in[i+1][j+1],grid_in[i+1][j]])
            rectangles.append(get_image_from_boxes(image_in,boxes[i*10+j]))
    return rectangles


def create_image(rectangles_in):
    """
    Appends all cut and transformed rectangles into one resulting image
    :param rectangles_in: list: list of cut and transformed rectangles
    :return: PIL.Image: resulting Image
    """

    lines=[]
    for i in range(10):
        lines.append([])
        lines[i]=Image.fromarray(np.hstack(tuple(rectangles_in[j*10+i] for j in range(10))))
    new_image=append_images(lines)
    return new_image
  
  
def append_images(images, direction='vertical',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images
    :param images: list: list of images to append
    :param direction: direction in which images should be appended
    :param bg_color: background color in RGB
    :param aligment: aligment of appended images
    :return: PIL.Image: resulting Image
    """

    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)
    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]
    return new_im


def Color(image_in, color_in):
    """
    Colors pixels in image in red
    :param image_in: np.ndarray: input image
    :param color_in: list of pixels to color
    :return: PIL.Image: resulting Image
    """

    i1=image_in.copy()
    i1[color_in]=[255,0,0]
    return transforms.ToPILImage()(i1)


def pixel_remap(image_in, mask_in):
    """
    Remaps image according to mask pixel by pixel
    :param image_in: np.ndarray: input image
    :param mask_in: input mask
    :return: np.ndarray: resulting Image
    """

    i1=np.ones((1500,1100,3),dtype=np.uint8)*200
    i0=image_in
    for i in range(1500):
        for j in range(1100):
            if ((mask_in[i][j][0]>0) and (mask_in[i][j][1]>0)):
                x=np.int(mask_in[i][j][0]*550-1)
                y=np.int(mask_in[i][j][1]*750-1)
                i1[y][x]=i0[i][j]
    return i1
