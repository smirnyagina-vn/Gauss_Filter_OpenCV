#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>

#define INPUT_IMAGE_NAME "C:/Users/Asus/Pictures/Практика/pics/pics/640catG.bmp"
#define OUTPUT_IMAGE_NAME "C:/Users/Asus/Pictures/Практика/pics/640catG_gaus.bmp"
#define MAX_KERNEL_SIZE 100
#define DEFAULT_SIGMA 0.566

using namespace std;
using namespace cv;

/////////////////////////////////ФУНКЦИИ///////////////////////////////////////////

double GaussianFunction(double x, double sigma);
double ConvertIndexToArgument(double x, double kernelSize);
double** GaussianKernelGenerator(double sigma, unsigned int kernelSize);
void GaussianFilter(Mat inputImage, int kernelSize, double sigma, string path);

///////////////////////////////////////////////////////////////////////////////////

double GaussianFunction(double x, double sigma) 
{
    const double x_as_sigmas = x / sigma;
    return exp(-0.5 * x_as_sigmas * x_as_sigmas) / (sigma * sqrt(2.0 * M_PI));
}


double ConvertIndexToArgument(double x, double kernelSize)
{
    double index = (2.0 * x - kernelSize + 1)/kernelSize;//перевод индексов матрицы, начинающихся с нуля                                              
    return index;                                        //в индексы с диапазоном (-kernelSize/2;kernelsize/2)
}


double** GaussianKernelGenerator(double sigma, unsigned int kernelSize)
{
    double matrixDiv = 0.0;//сумма всех элементов ядра для нормировки

    double** gaussKernel = new double* [kernelSize];
    for (int i = 0; i < kernelSize; i++)
        gaussKernel[i] = new double[kernelSize];

    //генерация двумерного ядра с функцией Гаусса
    for (int i = 0; i < kernelSize; i++)
    {
        for (int j = 0; j < kernelSize; j++)
        {
            //для получения матрицы нужно высчитать функции по строками столбцам
            gaussKernel[i][j] = GaussianFunction(ConvertIndexToArgument(i, kernelSize), sigma)
                * GaussianFunction(ConvertIndexToArgument(j, kernelSize), sigma);
            //cout << gaussKernel[i][j] << " ";
            matrixDiv += gaussKernel[i][j];
        }
       //cout << endl;
    }
    //cout << endl;

    cout << "Gaussian kernel " << kernelSize << "x" << kernelSize << " with sigma = ";
    cout << fixed;//для вывода
    cout.precision(3);//трёх знаков после запятой
    cout << sigma << ":" << endl;

    cout << "********************************" << endl;
    for (int i = 0; i < kernelSize; i++)
    {
        cout << " ";
        for (int j = 0; j < kernelSize; j++)
        {
            gaussKernel[i][j] /= matrixDiv;//нормировка матрицы
            cout << fixed;
            cout.precision(5);
            cout << gaussKernel[i][j] << " ";
        }
        cout << endl;
    }
    cout << "********************************" << endl;

    return gaussKernel;
}


void GaussianFilter(Mat inputImage, int kernelSize, double sigma, string path)
{
    double** gaussKernel = GaussianKernelGenerator(sigma, kernelSize);

    Mat bufImage, blurredImage;

    unsigned int border = kernelSize / 2;

    //добавляем границы для изображение с копированием пикселей, чтобы ядро не выходило за границы изображения
    copyMakeBorder(inputImage, bufImage, border, border, border, border, BORDER_REPLICATE);

    imshow("Input Image", inputImage);

    blurredImage = bufImage;

    //Свёртка изображения
   double sum = 0;
    for (int matY = 0; matY < bufImage.rows  - kernelSize; matY++)
    {
        for (int matX = 0; matX < bufImage.cols - kernelSize; matX++)
        {
            for (int kernelY = 0; kernelY < kernelSize; kernelY++)
            {
                for (int kernelX = 0; kernelX < kernelSize; kernelX++)
                {
                    sum += gaussKernel[kernelY][kernelX] * bufImage.at<uchar>(matY + kernelY, matX + kernelX);
                }
            }
            //Отслеживаем, чтобы цвет был в дозволенном диапазоне
            if (sum > 255) blurredImage.at<uchar>(matY + kernelSize / 2, matX + kernelSize / 2) = 255;
            else if (sum < 0) blurredImage.at<uchar>(matY + kernelSize / 2, matX + kernelSize / 2) = 0;
            else blurredImage.at<uchar>(matY + kernelSize / 2, matX + kernelSize / 2) = sum;
            sum = 0;
        }
    }

    //imshow("Output Image", blurredImage);

    Mat croppedImage = blurredImage(Rect(border, border, inputImage.cols, inputImage.rows));

    imshow("Cropped Image", croppedImage);

    imwrite(path, croppedImage);

}


int main(int argc, char** argv) {

    Mat image;

    unsigned int kernelSize;

    /*string filePath;
    cout << "Enter the path to the file: ";
    cin >> filePath;*/

    image = imread(INPUT_IMAGE_NAME, IMREAD_GRAYSCALE);

    if (!image.data) {
        cout << "Error loading image" << "\n";
        return -1;
    }
    
    cout << "Enter the size of kernel: ";
    cin >> kernelSize;
    if (kernelSize > MAX_KERNEL_SIZE || !cin.good())
    {
        cout << "Wrong input";
        return -1;
    }

    GaussianFilter(image, kernelSize, DEFAULT_SIGMA, OUTPUT_IMAGE_NAME);

    cout << "Image have been blurred" << endl;
    
    waitKey(0);

    return 0;
}