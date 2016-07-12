import numpy
import PIL.Image as Image
from skimage import color

    # Comments :
    #
    # DATABASE Checking : Clef2013
    #
    #
    # CieLab seems to be the worst of all colorspaces so far
    #
    #
    # RGB (-1 ,1 )  is the winner !!. (0 , 1 ) : 30-31 %   ( -1 ,1 ) 35-36 % with certain parameters
    # HSV (-1 ,1 )  is the winner !!. (0 , 1 ) : 33 %   ( -1 ,1 ) 36 % with certain parameters
    # CieLab in range ( 0 ,1 ) and ( -1 ,1  ) have the same results with little difference  (  35-36 % both  )
    #
    #
    # HSV , CieLab, Grayscale have better results if divided by 255
    # Grayscale with 1 channel better results than  3 channels copied
    #


    # RGB : RANGE 0-255
    #
    #
    # HSV Range : For HSV,
    #
    # Hue range is [0,179],
    #
    # Saturation range is [0,255] and
    #
    # Value range is [0,255].
    #
    #
    #CieLab : L A B
    #
    # L in [0, 100]
    #
    # A in [-86.185, 98,254]  128 - 128 kai sta 2
    #
    # B in [-107.863, 94.482] 128 - 128 kai sta 2
    #
    #
    # XYZ
    # X from 0 to  95.047
    # Y from 0 to 100.000
    # Z from 0 to 108.883
    #
    #


class ImageProcessing(object):


    def __init__(self):
        self.imagesDType = numpy.float32

    def rgb2rgbcie(self,imageArray):
        return color.rgb2rgbcie(imageArray)

    def rgbcie2rgb(self, imageArray):
        return color.rgbcie2rgb(imageArray)

    def rgb2luv(self,imageArray):
        return color.rgb2luv(imageArray)

    def luv2rgb(self, imageArray):
        return color.luv2rgb(imageArray)

    def rgb2xyz(self,imageArray):
        return color.rgb2xyz(imageArray)

    def xyz2rgb(self, imageArray):
        return color.xyz2rgb(imageArray)

    def rgb2hsv(self, imageArray):
        return color.rgb2hsv(imageArray)

    def hsv2rgb(self, imageArray):
        return color.hsv2rgb(imageArray)

    def rgb2CieLab(self, imageArray):
        return color.rgb2lab(imageArray / 255)

    def CieLab2rgb(self, imageArray):
        return color.lab2rgb(imageArray) * 255

    def rgb2gray(self, imageArray):
        return color.rgb2gray(imageArray)

    def gray2rgb(self, imageArray):
        return color.gray2rgb(imageArray)

    def convertArrayToRange(self,array,inputMax, inputMin,outputMax,outputMin):
        # normalize the input, in place
        array -= inputMin
        array /= inputMax

        # Convert the 0-1 range into a value in the right range.
        array *= outputMax - outputMin
        array += outputMin
        return array

    def repeatArrayOverNewDim(self, array, repeats, axis):
        return numpy.repeat(array[:, :, numpy.newaxis], repeats=repeats, axis=axis, )

    def getImageArray(self, imagePath, imageSize=(256, 256), resize=True, convertion=None, imageChannels=1,
                      preprocessImages=False,Normalize=True,NormalizeRange= ( 0 , 1 )):
        # RGB: Default
        if convertion not in [None, 'RGB', 'HSV', 'Grayscale', 'CieLab' , 'XYZ' , 'RGBCIE' , 'LUV' ]:
            raise ValueError('Not supported Color Type %r!' % convertion)

        img = self.openImage(imagePath=imagePath)
        if resize:
            img = self.resizeImage(image=img, size=imageSize)

        imageArray = self.Image2array(img=img)

        # if image is grayscale convert to (width,height,3 channels)
        if (imageArray.ndim == 2):
            # convert image ( repeat 3 times in third axis)
            imageArray = self.repeatArrayOverNewDim(array=imageArray, repeats=3, axis=2)

        if convertion == "Grayscale":
             # Images of type float must be between -1 and 1
            imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
                                                  outputMax=NormalizeRange[1],outputMin=NormalizeRange[0])

            imageArray = self.rgb2gray(imageArray)
            # add 3th dimension
            if imageChannels == 3:
                imageArray = self.repeatArrayOverNewDim(array=imageArray, repeats=3, axis=2)
            elif imageChannels == 1:
                imageArray = numpy.expand_dims(a=imageArray, axis=3)
        elif convertion == "HSV":
            # Images of type float must be between -1 and 1
            # recommended
            # imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
            #                                       outputMax=-1,outputMin=1)
            imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
                                                  outputMax=NormalizeRange[1],outputMin=NormalizeRange[0])
            imageArray = self.rgb2hsv(imageArray)
        elif convertion == "CieLab":
            imageArray = self.rgb2CieLab(imageArray)
        elif convertion == "XYZ":
            # Images of type float must be between -1 and 1
            # recommended
            # imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
            #                                    outputMax=1,outputMin=0)
            imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
                                                  outputMax=NormalizeRange[1],outputMin=NormalizeRange[0])
            imageArray = self.rgb2xyz(imageArray)

        elif convertion == "RGBCIE":
            # Images of type float must be between -1 and 1
            # recommended
            # imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
            #                                    outputMax=1,outputMin=-1)
            imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
                                                  outputMax=NormalizeRange[1],outputMin=NormalizeRange[0])
            imageArray = self.rgb2rgbcie(imageArray)
        elif convertion == "LUV":
            # Images of type float must be between -1 and 1
            # recommended
            # imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
            #                                    outputMax=1,outputMin=0)
            imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
                                                  outputMax=NormalizeRange[1],outputMin=NormalizeRange[0])
            imageArray = self.rgb2luv(imageArray)

        # preprocess image
        if preprocessImages:
            imageArray = self.preprocess_image(imageArray)

        # normalize image to (0,1)
        if Normalize:
            # depending on color convert to (0,1) or (-1,1)
            if convertion in [ None , 'RGB' ] :
                # recommended
                # imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
                #                                       outputMax=-1,outputMin=1)
                imageArray = self.convertArrayToRange(array=imageArray ,inputMax=255,inputMin=0,
                                                  outputMax=NormalizeRange[1],outputMin=NormalizeRange[0])
            elif convertion == 'CieLab':
                # recommended
                # imageArray = self.convertArrayToRange(array=imageArray ,inputMax=128,inputMin=-128,
                #                                       outputMax=1,outputMin=0)
                imageArray = self.convertArrayToRange(array=imageArray ,inputMax=128,inputMin=-128,
                                                  outputMax=NormalizeRange[1],outputMin=NormalizeRange[0])


        return imageArray

    def openImage(self, imagePath):
        return Image.open(imagePath)

    def Image2array(self, img):
        return numpy.array(img, dtype=self.imagesDType )

    def array2Image(self, array):
        return Image.fromarray(array)

    def saveImage(self, image, path):
        image.save(path)

    def flatten_matrix(self, matrix):
        vector = matrix.flatten(1)
        vector = vector.reshape(1, len(vector))
        return vector

    #  MULTIPLY ZCA MATRIX WITH TEST
    def zca_whitening(self, inputs):
        sigma = numpy.dot(inputs, inputs.T) / inputs.shape[1]  # Correlation matrix
        U, S, V = numpy.linalg.svd(sigma)  # Singular Value Decomposition
        epsilon = 0.1  # Whitening constant, it prevents division by zero
        ZCAMatrix = numpy.dot(numpy.dot(U, numpy.diag(1.0 / numpy.sqrt(numpy.diag(S) + epsilon))),
                              U.T)  # ZCA Whitening matrix
        return [numpy.dot(ZCAMatrix, inputs), ZCAMatrix]  # Data whitening

    def resizeImage(self, image, size=(256, 256), filter=Image.ANTIALIAS):
        # Filter Modes from lower to higher quality:
        # BILINEAR (linear interpolation in a 2x2 environment),
        # BICUBIC (cubic spline interpolation in a 4x4 environment),
        # ANTIALIAS (a high-quality downsampling filter)
        #
        #
        # Use ANTIALIAS unless speed is an issue ( Benchmarking showed that ANTIALIAS is slighlty slower)
        return image.resize(size, filter)

    def ImageMeanSubtraction(self, image):
        # mean subtraction
        image -= numpy.mean(image)
        return image

    def ImageSTD(self, imageArray):
        imageArray -= numpy.std(imageArray)
        return imageArray

    def ImageSmoothing(self, imageArray):
        from scipy import ndimage
        imageArray /= 255
        sigma = (14 * imageArray.shape[0] / 255.0, 14 * imageArray.shape[1] / 255.0, 0)
        imageArray = ndimage.gaussian_filter(imageArray, sigma=sigma, order=0)
        imageArray *= 255

        return imageArray

    # util function to open, resize and format pictures into appropriate tensors
    def preprocess_image(self, imageArray):
        # mean subtraction
        imageArray = self.ImageMeanSubtraction(imageArray)
        # normalization
        imageArray = self.ImageSTD(imageArray)
        # smooth image
        imageArray = self.ImageSmoothing(imageArray)

        imageArray = imageArray.transpose((2, 0, 1)).astype('float32')
        imageArray[:, :, 0] -= 103.939
        imageArray[:, :, 1] -= 116.779
        imageArray[:, :, 2] -= 123.68
        return imageArray

    # util function to convert a tensor into a valid image
    def deprocess_image(self, imageArray):
        imageArray[:, :, 0] += 103.939
        imageArray[:, :, 1] += 116.779
        imageArray[:, :, 2] += 123.68
        imageArray = imageArray.transpose((1, 2, 0))
        imageArray = numpy.clip(imageArray, 0, 255).astype('uint8')
        return imageArray

    def main(self):

        imagePath = "C:\\Users\\l.valavanis\\Desktop\\12.jpg"
        outPath = "C:\\Users\\l.valavanis\\Desktop\\11232.png"

        imagePath = '/home/leonidas/Desktop/relationship.jpg'
        image = self.openImage(imagePath)
        array =  self.Image2array(image)

        # print array
        array = numpy.array(array , dtype=numpy.float32)

        xyzArray = self.rgb2xyz(array)
        print xyzArray
        # # CIELAB to 0,1
        # # MIN MAX :  -107.863 ,100
        #
        # CieLAbarray = self.rgb2CieLab(array)
        # # print CieLAbarray
        # print CieLAbarray.shape
        # print numpy.max(CieLAbarray,axis=2)
        # array = self.convertArrayToRange(array=array ,inputMax=128,inputMin=-128,outputMax=1,outputMin=0)
        # print array


        # HSV TO 0,1
        #
        # FIRST CONVERT RGB  TO , ( -1 , 1 )
        # array = self.convertArrayToRange(array=array ,inputMax=255,inputMin=0,outputMax=1,outputMin=0)
        # print array
        # HSVarray = self.rgb2hsv(array)
        # print HSVarray
        #
        # array = self.hsv2rgb(HSVarray)
        # print array
        # array = self.convertArrayToRange(array=array ,inputMax=1,inputMin=-1,outputMax=255,outputMin=0)
        #
        # print array



        # RGB TO 0,1
        # normHSVArray

        # normRGBArray = self.convertArrayToRange(array=array ,inputMax=255,inputMin=0,outputMax=1,outputMin=0)
        # print "------"
        # print normRGBArray


        # array = self.preprocess_image(imagePath)
        # array = self.getImageArray(imagePath=imagePath,imageSize=(4,4),convertion="CieLab")
        # img2 = self.array2Image(array)



        # img2 = self.resizeImage(img2,filter = Image.BILINEAR)
        # self.saveImage(imaage,outPath)


if __name__ == '__main__':

    ImageProcessing().main()
