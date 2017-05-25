#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

// 添加这两个是因为win32程序  
//#pragma comment( lib, "vfw32.lib" )  
//#pragma comment( lib, "comctl32.lib" )  

cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
	int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
	int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
	if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
	{
		cv::Point2f pt;
		pt.x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / d;
		pt.y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / d;
		//-10 is a threshold, the POI can be off by at most 10 pixels
		if (pt.x<min(x1, x2) - 10 || pt.x>max(x1, x2) + 10 || pt.y<min(y1, y2) - 10 || pt.y>max(y1, y2) + 10) {
			return Point2f(-1, -1);
		}
		if (pt.x<min(x3, x4) - 10 || pt.x>max(x3, x4) + 10 || pt.y<min(y3, y4) - 10 || pt.y>max(y3, y4) + 10) {
			return Point2f(-1, -1);
		}
		return pt;
	}
	else
		return cv::Point2f(-1, -1);
}

bool comparator(Point2f a, Point2f b) {
	return a.x < b.x;
}
bool comparator2(vector<cv::Point2f> a, vector<cv::Point2f> b) {
	return a.size() > b.size();
}
bool comparatory(Point2f a, Point2f b) {
	return a.y < b.y;
}
void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
	std::vector<cv::Point2f> top, bot;
	for (int i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}
	sort(top.begin(), top.end(), comparator);
	sort(bot.begin(), bot.end(), comparator);
	cv::Point2f tl = top[0];
	cv::Point2f tr = top[top.size() - 1];
	cv::Point2f bl = bot[0];
	cv::Point2f br = bot[bot.size() - 1];
	corners.clear();
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);
}

cv::Mat debugSquares(std::vector<std::vector<cv::Point> > squares, cv::Mat image)
{
	for (int i = 0; i < squares.size(); i++) {
		// draw contour
		cv::drawContours(image, squares, i, cv::Scalar(255, 0, 0), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());

		// draw bounding rect
		cv::Rect rect = boundingRect(cv::Mat(squares[i]));
		cv::rectangle(image, rect.tl(), rect.br(), cv::Scalar(0, 255, 0), 2, 8, 0);

		// draw rotated rect
		cv::RotatedRect minRect = minAreaRect(cv::Mat(squares[i]));
		cv::Point2f rect_points[4];
		minRect.points(rect_points);
		for (int j = 0; j < 4; j++) {
			cv::line(image, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, 8); // blue
		}
	}

	return image;
}
cv::Point2f GenerateIndexedRectangles(std::vector<cv::Point2f> corners, int bias)
{
	cv::Point2f minLength(FLT_MAX, FLT_MAX);
	vector<float> xcoord;
	vector<float> ycoord;
	sort(corners.begin(), corners.end(), comparator);
	int counter = 0;
	float accumlator = 0;
	float temp = corners[0].x;
	for (int i = 0; i < corners.size(); i++)
	{
		if (abs((int)(corners[i].x - temp)) < bias)
		{
			accumlator += corners[i].x;
			counter++;
			temp = corners[i].x;
		}
		else
		{
			temp = corners[i].x;
			if (counter >= 5) {
				xcoord.push_back(accumlator / counter);
				accumlator = 0;
				counter = 0;
			}
		}
		cout << corners[i].x << endl;
	}
	if (counter >= 5) {
		xcoord.push_back(accumlator / counter);
	}
	sort(corners.begin(), corners.end(), comparatory);
	counter = 0;
	accumlator = 0;
	temp = corners[0].y;
	for (int i = 0; i < corners.size(); i++)
	{
		if (abs((int)(corners[i].y - temp)) < bias)
		{
			accumlator += corners[i].y;
			counter++;
			temp = corners[i].y;
		}
		else
		{
			if (counter >= 3) {
				temp = corners[i].y;
				ycoord.push_back(accumlator / counter);
				accumlator = 0;
				counter = 0;
			}
		}
		
	}
	if (counter >= 3) {
		ycoord.push_back(accumlator / counter);
	}

	vector<float> ydist;
	vector<float> xdist;
	for (int i = 1; i < xcoord.size(); i++) {
		minLength.x = minLength.x < xcoord[i] - xcoord[i - 1] ? minLength.x : xcoord[i] - xcoord[i - 1];
		xdist.push_back(xcoord[i] - xcoord[i - 1]);
		cout << xcoord[i] << endl;
	}
	for (int i = 1; i < ycoord.size(); i++) {
		minLength.y = minLength.y < ycoord[i] - ycoord[i - 1] ? minLength.y : ycoord[i] - ycoord[i - 1];
		cout << ycoord[i] << endl;
		ydist.push_back(ycoord[i] - ycoord[i - 1]);
	}
	/*sort(xcoord.begin(), xcoord.end());
	minLength.x = xcoord[2];*/
	sort(xdist.begin(), xdist.end());
	minLength.x = xdist[2];
	sort(ydist.begin(), ydist.end());
	minLength.y = ydist[2];
		
	//minLength.x -= 10;
	//minLength.y -= 10;

	cout << minLength.x << " " << minLength.y << endl;
	//minLength.y = 75;
	////minLength.y = 85;
	//minLength.x = 138;


	return minLength;

}

int main(int argc, char* argv[])
{

	string filenmae(argv[1]);
	Mat img = imread(filenmae, CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img = imread("F:\\PLM\\PLMUT\\Debug\\MDS00001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	if (img.empty())
	{
		cout << "error";
		return -1;
	}
	cv::resize(img, img, Size(1248, 1728), 0, 0);


	cv::Size size(3, 3);
	//cv::GaussianBlur(img,img,size,0);  
	adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 75, 20);
	cv::bitwise_not(img, img);


	vector<Vec4i> lines;
	HoughLinesP(img, lines, 1, CV_PI / 180, 30, 50, 12);
	//HoughLinesP(img, lines, 1, CV_PI/180, 80, 100, 12); 
	//HoughLinesP(img, lines, 50, CV_PI/150, 80, 500, 60); 

	//  cv::bitwise_not(img, img);  

	cvtColor(img, img, CV_GRAY2RGB);
	for (int i = 0; i < lines.size(); i++) {
		cv::Point   pt1(lines[i][0], lines[i][1]);
		cv::Point   pt2(lines[i][2], lines[i][3]);
		//cv::line( img, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0), 1, 8 ); // blue

		//cv::circle( img, Point(lines[i][0], lines[i][1]),8, cv::Scalar(0,0,255), 1, 8 ); // blue
		//cv::circle( img,  Point(lines[i][2], lines[i][3]), 8, cv::Scalar(0,0,255), 1, 8 ); // blue
	}


	int* poly = new int[lines.size()];
	for (int i = 0; i < lines.size(); i++)poly[i] = -1;
	int curPoly = 0;
	vector<vector<cv::Point2f> > corners;
	for (int i = 0; i < lines.size(); i++)
	{
		for (int j = i + 1; j < lines.size(); j++)
		{

			cv::Point2f pt = computeIntersect(lines[i], lines[j]);
			if (pt.x >= 0 && pt.y >= 0 && pt.x < img.size().width&&pt.y < img.size().height) {

				if (poly[i] == -1 && poly[j] == -1) {
					vector<Point2f> v;
					v.push_back(pt);
					corners.push_back(v);
					poly[i] = curPoly;
					poly[j] = curPoly;
					curPoly++;
					continue;
				}
				if (poly[i] == -1 && poly[j] >= 0) {
					bool exist = false;
					for (int k = 0; k < corners[poly[j]].size(); k++)
					{
						double tmp = sqrt((corners[poly[j]][k].x - pt.x) * (corners[poly[j]][k].x - pt.x)
							+ (corners[poly[j]][k].y - pt.y) *(corners[poly[j]][k].y - pt.y));
						cout << tmp << endl;
						if (sqrt((corners[poly[j]][k].x - pt.x) * (corners[poly[j]][k].x - pt.x)
							+ (corners[poly[j]][k].y - pt.y) *(corners[poly[j]][k].y - pt.y)) < 10)
						{// ignore points.
							exist = true;
							break;
						}
					}
					if (!exist)
						corners[poly[j]].push_back(pt);
					poly[i] = poly[j];

					continue;
				}
				if (poly[i] >= 0 && poly[j] == -1) {
					bool exist = false;
					for (int k = 0; k < corners[poly[i]].size(); k++)
					{
						double tmp = sqrt((corners[poly[i]][k].x - pt.x) * (corners[poly[i]][k].x - pt.x)
							+ (corners[poly[i]][k].y - pt.y) *(corners[poly[i]][k].y - pt.y));
						cout << tmp << endl;
						if (sqrt((corners[poly[i]][k].x - pt.x) * (corners[poly[i]][k].x - pt.x)
							+ (corners[poly[i]][k].y - pt.y) *(corners[poly[i]][k].y - pt.y)) < 10)
						{// ignore points.
							exist = true;
							break;
						}
					}
					if (!exist)

						corners[poly[i]].push_back(pt);
					poly[j] = poly[i];

					continue;
				}
				if (poly[i] >= 0 && poly[j] >= 0) {
					if (poly[i] == poly[j]) {
						bool exist = false;
						for (int k = 0; k < corners[poly[i]].size(); k++)
						{
							double tmp = sqrt((corners[poly[i]][k].x - pt.x) * (corners[poly[i]][k].x - pt.x)
								+ (corners[poly[i]][k].y - pt.y) *(corners[poly[i]][k].y - pt.y));
							cout << tmp << endl;
							if (sqrt((corners[poly[i]][k].x - pt.x) * (corners[poly[i]][k].x - pt.x)
								+ (corners[poly[i]][k].y - pt.y) *(corners[poly[i]][k].y - pt.y)) < 10)
							{// ignore points.
								exist = true;
								break;
							}
						}
						if (!exist)
							corners[poly[i]].push_back(pt);
						continue;
					}

					for (int k = 0; k < corners[poly[j]].size(); k++) {
						corners[poly[i]].push_back(corners[poly[j]][k]);
					}

					corners[poly[j]].clear();
					poly[j] = poly[i];
					continue;
				}
			}
		}
	}
	vector<float> distance;
	for (int i = 0; i < corners.size(); i++) {
		cv::Point2f center(0, 0);
		if (corners[i].size() < 4)continue;
		for (int j = 0; j < corners[i].size(); j++) {
			center += corners[i][j];
			distance.push_back(sqrt(corners[i][j].x * corners[i][j].x + corners[i][j].y * corners[i][j].y));
			cv::circle(img, corners[i][j], 8, cv::Scalar(0, 255, 0), 1, 8); // blue
																			//cout << corners[i].size() << endl;
			cout << corners[i][j] << endl;
		}
		//    sortCorners(corners[i], center);  
	}
	//imshow("xx的靓照", img);
	sort(distance.begin(), distance.end());
	for (int i = 0; i < distance.size(); i++)
	{
		//cout << distance[i] << endl;
	}

	int count = 0;
	sort(corners.begin(), corners.end(), comparator2);
	for (int i = 0; i < 1; i++) {

		if (corners[i].size() < 14)continue;
		size_t tsize = corners[i].size();

		cv::Point2f minLength;
		
		if (argc > 2) {
			minLength.x = atoi(argv[2]);
			minLength.y = atoi(argv[3]);
		}
		//else {
			minLength = GenerateIndexedRectangles(corners[i], 10);
		//}
		for (int j = 0; j < corners[i].size(); j++) {
			count++;
			vector<cv::Point2f> tcorners;
			int xbias = 2;
			int ybias = 2;
			tcorners.push_back(Point2f(corners[i][j].x + xbias, corners[i][j].y + ybias));
			tcorners.push_back(Point2f(corners[i][j].x + minLength.x - xbias - 1, corners[i][j].y + ybias));
			tcorners.push_back(Point2f(corners[i][j].x + minLength.x - xbias - 1, corners[i][j].y + minLength.y - ybias));
			tcorners.push_back(Point2f(corners[i][j].x + xbias, corners[i][j].y + minLength.y - ybias));

			Rect r = boundingRect(tcorners);
			//Rect r(corners[i][j].x, corners[i][j].y, minLength.x, minLength.y);
			cout << r.area() << endl;
			if (r.area() < 5000)continue;
			int ares = r.area();
			//if (r.area() > 10000000) continue;

			// Define the destination image  
			cv::Mat quad = cv::Mat::zeros(r.height, r.width, CV_8UC3);
			// Corners of the destination image  
			std::vector<cv::Point2f> quad_pts;
			quad_pts.push_back(cv::Point2f(0, 0));
			quad_pts.push_back(cv::Point2f(quad.cols, 0));
			quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
			quad_pts.push_back(cv::Point2f(0, quad.rows));
			// Get transformation matrix  
			cv::Mat transmtx = cv::getPerspectiveTransform(tcorners, quad_pts);
			// Apply perspective transformation  
			cv::warpPerspective(img, quad, transmtx, quad.size());
			stringstream ss;

			Mat hsv;
			//	cvtColor(quad, hsv, CV_BGR2HSV);
			cvtColor(quad, hsv, CV_BGR2GRAY);


			//	    Mat bw;
			//	inRange(hsv, Scalar(10, 10, 10), Scalar(250, 250, 250), bw);
			vector<vector<Point> > contours;
			findContours(hsv, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);


			Mat dst = Mat::zeros(quad.size(), quad.type());
			double totalArea = 0;

			vector<Point> points;

			for (int k = 0; k < contours.size(); k++)
			{
				double tmpArea = contourArea(contours[k]);
				if (tmpArea > 10) {
					points.insert(points.end(), contours[k].begin(), contours[k].end());
					totalArea += tmpArea;
				}
				cout << tmpArea << " ";

			}
			cout << totalArea << endl;
			if (totalArea > 100) {
				// drawContours(dst, contours, -1, Scalar::all(167), CV_FILLED);
				// imshow("src", quad);
				// imshow("dst", dst);
				//dst &= quad;

				cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
				double angle = box.angle;
				if (angle < -45.)
					angle += 90.;
				cout << "angle" << angle << endl;
				if (angle > 5) angle = 5;
				if (angle < -5) angle = -5;


				cv::Point2f vertices[4];
				box.points(vertices);
				//for(int i = 0; i < 4; ++i)
				//cv::line(quad, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);
				//				imshow("src", quad);


				//Rotation
				cv::Mat rot_mat = cv::getRotationMatrix2D(box.center, angle, 1);
				cv::Mat rotated;
				cv::warpAffine(quad, rotated, rot_mat, quad.size(), cv::INTER_CUBIC);

				//Cropping
				cv::Size box_size = box.size;
				if (box.angle < -45.)
					std::swap(box_size.width, box_size.height);
				cv::Mat cropped;
				cv::getRectSubPix(rotated, box_size, box.center, cropped);


				//Resize
				cv::Mat resized;
				cv::resize(cropped, resized, cv::Size(353, 141), 1, 0.5, CV_INTER_CUBIC);
				cv::bitwise_not(resized, resized);
				ss << i * 1000 + j << ".bmp";
				cv::cvtColor(resized, resized, CV_BGR2GRAY);
				cv::threshold(resized, resized, 108, 255, THRESH_BINARY);
				cv::imwrite(ss.str(), resized);


				//				imshow("dest", resized);
				//  imshow("dst", dst);
				waitKey(0);
			}

			// imshow(ss.str(), quad);  
		}
	}

	imshow("xx的靓照", img);
	waitKey();

	return 0;
}

