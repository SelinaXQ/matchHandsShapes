/*by Qian Xiang
Detect cloth, scissors, stone
*/

#include <iostream> 
#include <string> 
#include <iomanip> 
#include <sstream> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
using namespace cv;
using namespace std;


Mat GuestureDetect(Mat img) {

	//分别对3个模板进行处理 process three templates seperately
	Mat five = imread("template\\five.bmp");
	//imshow("five", five);
	cvtColor(five, five, CV_BGR2GRAY);
	threshold(five, five, 40, 255, CV_THRESH_BINARY);
	vector<vector<Point>> contours1;
	vector<Vec4i> hierarcy1;
	findContours(five, contours1, hierarcy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Mat  two = imread("template\\two.bmp");
	cvtColor(two, two, CV_BGR2GRAY);
	threshold(two, two, 40, 255, CV_THRESH_BINARY);
	vector<vector<Point>> contours2;
	vector<Vec4i> hierarcy2;
	findContours(two, contours2, hierarcy2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Mat  ten = imread("template\\ten.bmp");
	cvtColor(ten, ten, CV_BGR2GRAY);
	threshold(ten, ten, 40, 255, CV_THRESH_BINARY);
	vector<vector<Point>> contours3;
	vector<Vec4i> hierarcy3;
	findContours(ten, contours3, hierarcy3, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Mat  six = imread("template\\six.bmp");
	cvtColor(six, six, CV_BGR2GRAY);
	threshold(six, six, 40, 255, CV_THRESH_BINARY);
	vector<vector<Point>> contours4;
	vector<Vec4i> hierarcy4;
	findContours(six, contours4, hierarcy4, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Mat dst = img.clone();
	Mat imgHSV;//frame的HSV表现形式 HSV image
	Mat mask(img.rows, img.cols, CV_8UC1);  // 2值掩膜  binary mask
	Mat none = imread("result\\none.png");
	vector<vector<Point>> contours;//轮廓 contours
	vector< vector<Point> > filterContours; // 筛选后的轮廓  The filtered contour, the area should be larger than a threshold
	vector< Point > hull; // 凸包络的点集  convexity points used for drawing contours
	vector<Vec4i> hierarcy;
	medianBlur(img, img, 5);
	cvtColor(img, imgHSV, CV_BGR2HSV);
	Mat dstTemp1(img.rows, img.cols, CV_8UC1);
	Mat dstTemp2(img.rows, img.cols, CV_8UC1);
	// 对HSV空间进行量化，得到2值图像，亮的部分为手的形状  The HSV space was quantized to obtain the binary image, and the bright part was the shape of the hand
	inRange(imgHSV, Scalar(0, 30, 30), Scalar(40, 170, 256), dstTemp1);
	inRange(imgHSV, Scalar(156, 30, 30), Scalar(180, 170, 256), dstTemp2);
	bitwise_or(dstTemp1, dstTemp2, mask);//or 
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	// 形态学操作，去除噪声，并使手的边界更加清晰  Morphological operation to remove noise and make the boundary of the hand more clear
	erode(mask, mask, element);
	morphologyEx(mask, mask, MORPH_OPEN, element);
	dilate(mask, mask, element);
	morphologyEx(mask, mask, MORPH_CLOSE, element);
	img.copyTo(dst, mask);
	contours.clear();
	hierarcy.clear();
	filterContours.clear();



	findContours(mask, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); //查找轮廓 findcontours
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (fabs(contourArea(Mat(contours[i]))) > 8000) //判断手面积阈值  Judge the threshold of hand area
		{
			filterContours.push_back(contours[i]);
		}
	}
	// 画轮廓  draw contours, easy to watch
	if (filterContours.size() > 0)
	{

		drawContours(dst, filterContours, -1, Scalar(255, 0, 255), 3/*, 8, hierarchy*/);

		for (size_t j = 0; j < filterContours.size(); j++)
		{
			convexHull(Mat(filterContours[j]), hull, true);
			int hullcount = (int)hull.size();
			for (int i = 0; i < hullcount - 1; i++)
			{
				line(dst, hull[i + 1], hull[i], Scalar(255, 255, 255), 2, CV_AA);//白色  white       
			
			}

			line(dst, hull[hullcount - 1], hull[0], Scalar(0, 255, 0), 2, CV_AA); 

		}
	}

    cout << "Contours size:"<<filterContours.size() << endl;
	if (filterContours.size() == NULL) {
		
		imshow("result0", none);
		imshow("result1", none);
	}
	else {
		if (filterContours.size() == 1) {
			double matchrate1 = matchShapes(contours1[0], filterContours[0], CV_CONTOURS_MATCH_I1, 0.0);//matchShapes: The lower the result, the better match it is
			//cout << "index1=" << i << "---" << setiosflags(ios::fixed) << matchrate1 << endl;
			double matchrate2 = matchShapes(contours2[0], filterContours[0], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index2=" << i << "---" << setiosflags(ios::fixed) << matchrate2 << endl;
			double matchrate3 = matchShapes(contours3[0], filterContours[0], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index3=" << i << "---" << setiosflags(ios::fixed) << matchrate3 << endl;
			double matchrate4 = matchShapes(contours4[0], filterContours[0], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index4=" << i << "---" << setiosflags(ios::fixed) << matchrate3 << endl;
			if (matchrate1 < matchrate2 && matchrate1 < matchrate3 && matchrate1 < matchrate4)
			{
				Mat result = imread("result\\five.png");
				imshow("result0", result);
				imshow("result1", none);
				//putText(dst, "cloth", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate2 < matchrate1 && matchrate2 < matchrate3 && matchrate2 < matchrate4)
			{
				Mat result = imread("result\\two.png");
				imshow("result0", result);
				imshow("result1", none);
				//putText(dst, "scissors", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate3 < matchrate1 && matchrate3 < matchrate2 && matchrate3 < matchrate4)
			{
				Mat result = imread("result\\ten.png");
				imshow("result0", result);
				imshow("result1", none);
				//putText(dst, "stone", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate4 < matchrate1 && matchrate4 < matchrate2 && matchrate4 < matchrate3)
			{
				Mat result = imread("result\\six.png");
				imshow("result0", result);
				imshow("result1", none);
				//putText(dst, "stone", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
		}
		else {
			//第一只手 the first hand
			double matchrate1 = matchShapes(contours1[0], filterContours[0], CV_CONTOURS_MATCH_I1, 0.0);//matchShapes: The lower the result, the better match it is
			//cout << "index1=" << i << "---" << setiosflags(ios::fixed) << matchrate1 << endl;
			double matchrate2 = matchShapes(contours2[0], filterContours[0], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index2=" << i << "---" << setiosflags(ios::fixed) << matchrate2 << endl;
			double matchrate3 = matchShapes(contours3[0], filterContours[0], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index3=" << i << "---" << setiosflags(ios::fixed) << matchrate3 << endl;
			double matchrate4 = matchShapes(contours4[0], filterContours[0], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index4=" << i << "---" << setiosflags(ios::fixed) << matchrate3 << endl;
			if (matchrate1 < matchrate2 && matchrate1 < matchrate3 && matchrate1 < matchrate4)
			{
				Mat result = imread("result\\five.png");
				imshow("result0", result);
				//putText(dst, "cloth", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate2 < matchrate1 && matchrate2 < matchrate3 && matchrate2 < matchrate4)
			{
				Mat result = imread("result\\two.png");
				imshow("result0", result);
				//putText(dst, "scissors", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate3 < matchrate1 && matchrate3 < matchrate2 && matchrate3 < matchrate4)
			{
				Mat result = imread("result\\ten.png");
				imshow("result0", result);
				//putText(dst, "stone", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate4 < matchrate1 && matchrate4 < matchrate2 && matchrate4 < matchrate3)
			{
				Mat result = imread("result\\six.png");
				imshow("result0", result);
				//putText(dst, "stone", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}

			//第二只手 the second hand
			double matchrate5 = matchShapes(contours1[0], filterContours[1], CV_CONTOURS_MATCH_I1, 0.0);//matchShapes: The lower the result, the better match it is
			//cout << "index1=" << i << "---" << setiosflags(ios::fixed) << matchrate1 << endl;
			double matchrate6 = matchShapes(contours2[0], filterContours[1], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index2=" << i << "---" << setiosflags(ios::fixed) << matchrate2 << endl;
			double matchrate7 = matchShapes(contours3[0], filterContours[1], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index3=" << i << "---" << setiosflags(ios::fixed) << matchrate3 << endl;
			double matchrate8 = matchShapes(contours4[0], filterContours[1], CV_CONTOURS_MATCH_I1, 0.0);
			//cout << "index4=" << i << "---" << setiosflags(ios::fixed) << matchrate3 << endl;
			if (matchrate5 < matchrate6 && matchrate5 < matchrate7 && matchrate5 < matchrate8)
			{
				Mat result = imread("result\\five.png");
				imshow("result1", result);
				//putText(dst, "cloth", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate6 < matchrate5 && matchrate6 < matchrate7 && matchrate6 < matchrate8)
			{
				Mat result = imread("result\\two.png");
				imshow("result1", result);
				//putText(dst, "scissors", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate7 < matchrate5 && matchrate7 < matchrate6 && matchrate7 < matchrate8)
			{
				Mat result = imread("result\\ten.png");
				imshow("result1", result);
				//putText(dst, "stone", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
			if (matchrate8 < matchrate5 && matchrate8 < matchrate6 && matchrate8 < matchrate7)
			{
				Mat result = imread("result\\six.png");
				imshow("result1", result);
				//putText(dst, "stone", Point(30, 70), CV_FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255), 2, 8);
			}
		}
	}

	return dst;

}
void main() {

	VideoCapture cap(0);
	if (!cap.isOpened()) //检查是否打开摄像头 check whether the camera is open
		return;
	Mat frame;
	Mat dst;
	while (1)
	{
		cap >> frame;
		if (!frame.empty())
		{
			dst = GuestureDetect( frame);
			imshow("result", dst);

			if (waitKey(30) == 27)
				break;
		}
		else
			break;
	}
	cap.release();
}