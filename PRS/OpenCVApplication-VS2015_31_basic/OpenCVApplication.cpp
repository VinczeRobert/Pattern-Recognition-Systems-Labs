// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include "time.h"
#include <random>

using namespace cv;


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

// Lab 1 : Least Mean Squares Line Fitting
std::vector<Point2f> readPointsFromFile() {
	int width = 500;
	int height = 500;
	int dimension = 2;
	Mat matrix = Mat(height, width, CV_8UC3);

	FILE* f = fopen("Images/lab1/points3.txt", "r");

	if (f == NULL) {
		perror("File couldn't be opened! \n");
		exit(-1);
	}

	int nrOfPoints;
	fscanf(f, "%d", &nrOfPoints);

	std::vector<Point2f> points; 

	for (int i = 0; i < nrOfPoints; i++) {
		float x, y;
		fscanf(f, "%f%f", &x, &y);
		points.push_back(Point2f(x, y));

		if (x <= width && y <= height) {
			circle(matrix, Point2d(x, y), 3, Scalar(255, 0, 0), -1);
		}
	}

	fclose(f);
	return points;
}

Mat pointsToImage(std::vector<Point2f> points) {
	int n = points.size();
	int width = 500;
	int height = 500;
	Mat img = Mat(width, height, CV_8UC3);

	for (int i = 0; i < n; i++) {
		if (points.at(i).x <= height && points.at(i).y <= width) {
			circle(img, Point2d(points.at(i).x, points.at(i).y), 3, Scalar(255, 0, 0), -1);
		}
	}

	return img;
}

void withNormalEcuation() {
	std::vector<Point2f> points = readPointsFromFile();
	int n = points.size();
	Mat teta = Mat(2, 1, CV_32F);
	Mat A = Mat(n, 2, CV_32F);
	Mat b = Mat(n, 1, CV_32F);

	for (int i = 0; i < n; i++) {
		A.at<float>(i, 1) = points.at(i).x;
		A.at<float>(i, 0) = 1;
		b.at<float>(i, 0) = points.at(i).y;
	}

	teta = (A.t()*A).inv()*A.t()*b;
	printf("teta0 = %f, teta1 = %f\n", teta.at<float>(0, 0), teta.at<float>(1, 0));

	Mat img = pointsToImage(points);
	line(img, Point2d(0, teta.at<float>(0, 0)), Point2d(500, teta.at<float>(0, 0) + 500 *
		teta.at<float>(1, 0)), Scalar(0, 0, 0));
	imshow("points", img);
	waitKey(0);
}

void calculateThetaWithGradientDescent() {
	std::vector<Point2f> points = readPointsFromFile();
	int n = points.size();
	Mat teta = Mat(2, 1, CV_32F);

	// init teta with random values
	teta.at<float>(0, 0) = rand() % n;
	teta.at<float>(1, 0) = rand() % n;

	float alpha = 0.000001; // learning rate
	Mat img = pointsToImage(points);

	float error = 100000;

	while (true)
	{
		// calculate gradient
		Mat gradient = Mat(2, 1, CV_32F);
		gradient.at<float>(0, 0) = 0;
		gradient.at<float>(1, 0) = 0;
		error = 0;

		for (int i = 0; i < n; i++)
		{
			gradient.at<float>(0, 0) += teta.at<float>(0, 0) + teta.at<float>(1, 0) * points.at(i).x - points.at(i).y;
			gradient.at<float>(1, 0) += (teta.at<float>(0, 0) + teta.at<float>(1, 0) * points.at(i).x - points.at(i).y) * points.at(i).x;
			error += pow(teta.at<float>(0, 0) + teta.at<float>(1, 0) * points.at(i).x - points.at(i).y, 2) / 2;
		}

		teta = teta - alpha * gradient;
		line(img, Point2d(0, teta.at<float>(0, 0)), Point2d(500, teta.at<float>(0, 0) + 500 * teta.at<float>(1, 0)), Scalar(0, 0, 0));

		imshow("points", img);
		printf("%f\n", error);
		waitKey(100);
	}
}

void calculateTheta() {
	std::vector<Point2f> points = readPointsFromFile();
	int n = points.size();
	float sumXY = 0;
	float sumX = 0;
	float sumY = 0;
	float sumX2 = 0;

	for (int i = 0; i < n; i++) {
		sumX += points.at(i).x;
		sumY += points.at(i).y;
		sumXY += points.at(i).x * points.at(i).y;
		sumX2 += points.at(i).x * points.at(i).x;
	}

		float teta1 = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
		float teta0 = (sumY - teta1 * sumX) / n;

		Mat img = pointsToImage(points);
		line(img, Point2d(0, teta0), Point2d(500, teta0 + 500 * teta1), Scalar(0, 0, 0));

		imshow("points", img);
		waitKey(0);
}

void lineFittingMethod2() {
	std::vector<Point2f> points = readPointsFromFile();
	int n = points.size();
	float sumXY = 0;
	float sumX = 0;
	float sumY = 0;
	float sumX2Y2 = 0;

	for (int i = 0; i < n; i++) {
		sumXY += points.at(i).x * points.at(i).y;
		sumX += points.at(i).x;
		sumY += points.at(i).y;
		sumX2Y2 += pow(points.at(i).y, 2) - pow(points.at(i).x, 2);
	}

	float beta = -0.5 * atan2(2 * sumXY - 2 / n * sumX * sumY, sumX2Y2 + 1 / n * pow(sumX, 2) - 1 / n *pow(sumY, 2));
	float ro = (cos(beta) * sumX + sin(beta) * sumY) / n;

	Mat img = pointsToImage(points);
	line(img, Point(0, ro / sin(beta)), Point(img.cols, (ro - img.cols * cos(beta)) / sin(beta)), Scalar(0, 0, 255));

	imshow("points", img);
	waitKey(0);
}

std::vector<Point> readPoints() {
	Mat img = imread("Images/lab2/points1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat inv;
	cv::bitwise_not(img, inv);
	std::vector<Point> points;
	cv::findNonZero(inv, points);
	return points;
}

void ransac() {
	Mat img = imread("Images/lab2/points5.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat inv;
	cv::bitwise_not(img, inv);
	std::vector<Point> points;
	cv::findNonZero(inv, points);

	float t = 10;
	float p = 0.99;
	float q = 0.3;
	float s = 2;

	int n = points.size();
	float T = n*q;
	float N = log(1 - p) / log(1 - pow(q, s));

	int inliner = 0;
	Point3d optimalLine = Point3d(0, 0, 0);

	srand(time(0));
	for (int j = 0; (j < N) && (inliner < T); j++) {
		int p1 = rand() % n;
		int p2 = rand() % n;

		Point3f line;
		line.x = points.at(p1).y - points.at(p2).y;
		line.y = points.at(p2).x - points.at(p1).x;
		line.z = points.at(p1).x * points.at(p2).y - points.at(p2).x * points.at(p1).y;

		float dist = 0;

		int nrValidPoints = 0;

		for each(Point p in points) {
			dist = abs(line.x * p.x + line.y * p.y + line.z) / sqrt(line.x * line.x + line.y * line.y);
			if (dist <= t) {
				nrValidPoints++;
			}
		}	
		if (nrValidPoints > inliner) {
			inliner = nrValidPoints;
			optimalLine = line;
		}
	}

	line(img, Point2d(0, -optimalLine.z / optimalLine.y), Point2d(500, (-optimalLine.z - optimalLine.x * 500) / optimalLine.y), Scalar(0, 0, 0));
	imshow("RANSAC", img);
	waitKey(0);
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Lab1 : Least Mean Squares Line Fitting - Normal Equation\n");
		printf(" 11 - Lab1 : Least Mean Squares Line Fitting - Find theta\n");
		printf(" 12 - Lab1 : Least Mean Squares Line Fitting - Find theta with gradient descent\n");
		printf(" 13 - Lab1 : Least Mean Squares Line Fitting - Method 2 \n");
		printf(" 14 - Lab2 : RANSAC Line Fitting \n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				withNormalEcuation();
				break;
			case 11:
				calculateTheta();
				break;
			case 12:
				calculateThetaWithGradientDescent();
				break;
			case 13:
				lineFittingMethod2();
				break;
			case 14:
				ransac();

		}
	}
	while (op!=0);
	return 0;
}