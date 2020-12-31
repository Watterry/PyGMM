import logging
import numpy as np
import matplotlib.pyplot as plt
import pylab
import numpy as np  
import cv2
import math

alpha = 0.01
beta = alpha/(1/4)
sig_init = 9

def normalization(data):
    norm = np.linalg.norm(data, ord=2, axis=1, keepdims=True)
    normal_array = data/(norm*2)

    return normal_array

class GMM:
    """
    The Gaussian model
    """
    def __init__(self):
        """
        init the Guassian model of current pixel, random the mean and sigmod value
        """
        self.mu = np.random.randint(0, 255)
        self.sig = sig_init   # based on experience

    def probability(self, x):
        np.exp(-np.power(x - self.mu, 2.) / (2 * np.power(self.sig, 2.)))
    
class MixGMM:
    """
    The Gaussian distributions of the adaptive mixture model
    """
    def __init__(self, C):
        """
        init the Guassian model of current pixel, random the mean and sigmod value
        Args:
            C: the amount of Guassian models
        """
        self.C = C
        self.GMMs = np.empty((1, C), dtype=object)
        for i in range(C):
            self.GMMs[0][i] = GMM()

        self.weights = np.zeros((1, C), dtype=float)
        self.weights[:,:] = 1/C

    def updateGMM(self, pixel):
        """
        Update GMM models
        Args:
            C: the amount of Guassian models
        Returns:
            True if the pixel is force ground, False if the pixel is background
        """
        noMatch = True
        maxGap = 0
        maxIdx = 0

        for i in range(self.C):
            mu = self.GMMs[0, i].mu
            sig = self.GMMs[0, i].sig

            temp = abs(pixel - mu)
            if (temp < (2.5 * sig)):
                # current model is matched
                noMatch = False
                self.weights[0, i] = (1-alpha)*self.weights[0, i] + alpha
                self.GMMs[0, i].mu = (1-beta)*self.GMMs[0, i].mu + beta*pixel
                # TODO: need more match calculation here
                self.GMMs[0, i].sig = math.sqrt( (1-beta)*pow(sig,2) + beta*pow((pixel - mu),2) )
            else:
                # current model is not matched
                self.weights[0, i] = (1-alpha)*self.weights[0, i]
            
            # save the least probable distribution
            if maxGap < temp:
                maxGap = temp
                maxIdx = i

        if noMatch:
            #replace the least probable distribution if no match of all models
            self.GMMs[0, maxIdx].mu = pixel
            self.GMMs[0, maxIdx].sig = sig_init

        # re-normalize weights
        new_weights = normalization(self.weights)
        self.weights = new_weights
        
        if noMatch:
            return True
        else:
            return False

class ImageGMM:
    """
    GMM module for image with width & height pixels
    """
    def __init__(self, C, width, height):
        """
        Args:
            C: the amount of a mixture of Gaussians
            width: the width of image
            height: the height of image
        """
        self.C = int(C)
        self.width = int(width)
        self.height = int(height)
        self.GMM_matrix = np.empty((self.height, self.width), dtype=object)

        for x in range(self.height):
            for y in range(self.width):
                self.GMM_matrix[x][y] = MixGMM(self.C)

    def trainGMM(self, frame):
        """
        input a video frame, and update the mixture of Guassians of background
        """
        self.updateModel(frame)

    def extractFront(self, frame):
        """
        abstract the front moving objects of current frame
        """
        result = self.updateModel(frame)
        return result

    def updateModel(self, frame):
        """
        Update the mixture Guassian model
        Args:
            frame: the input video frame
        Returns:
            
        """
        result = np.zeros((self.height, self.width), dtype=int).astype('uint8')
        for x in range(self.height):
            for y in range(self.width):
                pixel = frame[x, y]

                isFG = self.GMM_matrix[x][y].updateGMM(pixel)
                if (isFG):
                    result[x,y] = 255

        return result

def test():
    cap = cv2.VideoCapture('D:/video/test5.mp4')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logging.info('video info: %d, %d, %d', count, width, height)

    gray_value = np.zeros(count, np.uint8)
    RGB_value = np.zeros((3, count), np.uint8)

    pixel_pos_x = 80
    pixel_pos_y = 80

    index = 0
    img_Gmm = ImageGMM(4, width, height)

    out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('a','v','c','1'), fps, (width,height), False)

    while(cap.isOpened()):  
        logging.info("process frame index: %d", index)
        ret, frame = cap.read()
        if ret==False:
            break

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray_value[index] = gray_img[pixel_pos_x, pixel_pos_y]
        # RGB_value[0][index] = frame[pixel_pos_x, pixel_pos_y][0]
        # RGB_value[1][index] = frame[pixel_pos_x, pixel_pos_y][1]
        # RGB_value[2][index] = frame[pixel_pos_x, pixel_pos_y][2]

        if index<100:
            #train the background of Gaussians
            img_Gmm.trainGMM(gray_img)
        else:
            result = img_Gmm.extractFront(gray_img)
            out.write(result)
            #cv2.imshow('image',vis)
            #cv2.waitKey(0)

        index = index + 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("PyGmm.log", mode='w', encoding = "UTF-8"),
            logging.StreamHandler(),
        ]
    )
    logging.getLogger('matplotlib.font_manager').disabled = True

    test()