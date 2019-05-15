#include <opencv2\opencv.hpp>
#include <string.h>
using namespace cv;
using namespace std;

const int NUM_X = 500;
const int NUM_Y = 500;

Mat sobel_x = (Mat_<double>(3, 3) << 1.0, 0.0, -1.0,
								2.0, 0.0, -2.0,
								1.0, 0.0, -1.0);
Mat sobel_y = (Mat_<double>(3, 3) << 1.0, 2.0, 1.0,
								0.0, 0.0, 0.0,
								-1.0, -2.0, -1.0);
Mat laplace = (Mat_<double>(3, 3) << 0.0, 1.0, 0.0,
								1.0, -4.0, 1.0,
								0.0, 1.0, 0.0);
Mat robert = (Mat_<double>(2, 2) << -1.0, 0.0,
								0.0, 1.0);

struct node{
	int cont;
	int pre;
};


//改变权值，实现图片保护
Mat protect(Mat deal, Mat gray){
	Mat res;
	gray.copyTo(res);
	for(int i = 0; i < deal.cols; i++){
		for(int j = 0; j < deal.rows; j++){
			if((int)deal.at<Vec3b>(j, i)[0] > 250 && deal.at<Vec3b>(j, i)[1] > 250 && deal.at<Vec3b>(j, i)[2] > 250){
				res.at<uchar>(j, i) = 255;
			}
		}
	}
	return res;
}

//改变权值，实现对象移除
Mat remove_object(Mat deal, Mat gray){
	Mat res;
	gray.copyTo(res);
	for(int i = 0; i < deal.cols; i++){
		for(int j = 0; j < deal.rows; j++){
			if((int)deal.at<Vec3b>(j, i)[0] > 250 && deal.at<Vec3b>(j, i)[1] > 250 && deal.at<Vec3b>(j, i)[2] > 250){
				res.at<uchar>(j, i) = 0;
			}
			else{
				res.at<uchar>(j, i) += 5;
			}
		}
	}
	return res;
}

//删除一行
Mat del_row(Mat ini, int** line, int index){
	Mat res(ini.rows-1, ini.cols, ini.type());
	for(int i = 0; i < ini.cols; i++){
		int temp = line[index][i];
		for(int j = 0; j < ini.rows; j++){
			if(j < temp){
				res.at<Vec3b>(j, i)[0] = ini.at<Vec3b>(j, i)[0];
				res.at<Vec3b>(j, i)[1] = ini.at<Vec3b>(j, i)[1];
				res.at<Vec3b>(j, i)[2] = ini.at<Vec3b>(j, i)[2];
			}
			else if(j > temp){
				if(j == temp+1){
					res.at<Vec3b>(j-1, i)[0] = (ini.at<Vec3b>(j-1, i)[0] + ini.at<Vec3b>(j, i)[0])/2;
					res.at<Vec3b>(j-1, i)[1] = (ini.at<Vec3b>(j-1, i)[1] + ini.at<Vec3b>(j, i)[1])/2;
					res.at<Vec3b>(j-1, i)[2] = (ini.at<Vec3b>(j-1, i)[2] + ini.at<Vec3b>(j, i)[2])/2;
				}
				else{
					res.at<Vec3b>(j-1, i)[0] = ini.at<Vec3b>(j, i)[0];
					res.at<Vec3b>(j-1, i)[1] = ini.at<Vec3b>(j, i)[1];
					res.at<Vec3b>(j-1, i)[2] = ini.at<Vec3b>(j, i)[2];
				}
			}
		}
	}
	return res;
}

//删除一列
Mat del_col(Mat ini, int** line, int index){
	Mat res(ini.rows, ini.cols-1, ini.type());
	for(int i = 0; i < ini.rows; i++){
		int temp = line[index][i];
		for(int j = 0; j < ini.cols; j++){
			if(j < temp){
				res.at<Vec3b>(i, j)[0] = ini.at<Vec3b>(i, j)[0];
				res.at<Vec3b>(i, j)[1] = ini.at<Vec3b>(i, j)[1];
				res.at<Vec3b>(i, j)[2] = ini.at<Vec3b>(i, j)[2];
			}
			else if(j > temp){
				if(j == temp+1){
					res.at<Vec3b>(i, j-1)[0] = (ini.at<Vec3b>(i, j)[0] + ini.at<Vec3b>(i, j-1)[0])/2;
					res.at<Vec3b>(i, j-1)[1] = (ini.at<Vec3b>(i, j)[1] + ini.at<Vec3b>(i, j-1)[1])/2;
					res.at<Vec3b>(i, j-1)[2] = (ini.at<Vec3b>(i, j)[2] + ini.at<Vec3b>(i, j-1)[2])/2;
				}
				else{
					res.at<Vec3b>(i, j-1)[0] = ini.at<Vec3b>(i, j)[0];
					res.at<Vec3b>(i, j-1)[1] = ini.at<Vec3b>(i, j)[1];
					res.at<Vec3b>(i, j-1)[2] = ini.at<Vec3b>(i, j)[2];
				}
			}
		}
	}
	return res;
}

//增加一行
Mat add_row(Mat ini, int** line, int index){
	Mat res(ini.rows+1, ini.cols, ini.type());
	for(int i = 0; i < res.cols; i++){
		int temp = line[index][i];
		for(int j = 0; j < res.rows; j++){
			if(j < temp){
				res.at<Vec3b>(j, i)[0] = ini.at<Vec3b>(j, i)[0];
				res.at<Vec3b>(j, i)[1] = ini.at<Vec3b>(j, i)[1];
				res.at<Vec3b>(j, i)[2] = ini.at<Vec3b>(j, i)[2];
			}
			else if(j > temp){
				res.at<Vec3b>(j, i)[0] = ini.at<Vec3b>(j-1, i)[0];
				res.at<Vec3b>(j, i)[1] = ini.at<Vec3b>(j-1, i)[1];
				res.at<Vec3b>(j, i)[2] = ini.at<Vec3b>(j-1, i)[2];
			}
			else{
				if(j > 0){
					res.at<Vec3b>(j, i)[0] = (ini.at<Vec3b>(j, i)[0]+ini.at<Vec3b>(j-1, i)[0])/2;
					res.at<Vec3b>(j, i)[1] = (ini.at<Vec3b>(j, i)[1]+ini.at<Vec3b>(j-1, i)[1])/2;
					res.at<Vec3b>(j, i)[2] = (ini.at<Vec3b>(j, i)[2]+ini.at<Vec3b>(j-1, i)[2])/2;
				}
				else{
					res.at<Vec3b>(j, i)[0] = ini.at<Vec3b>(j, i)[0];
					res.at<Vec3b>(j, i)[1] = ini.at<Vec3b>(j, i)[1];
					res.at<Vec3b>(j, i)[2] = ini.at<Vec3b>(j, i)[2];
				}
			}
		}
	}
	return res;
}

//增加一列
Mat add_col(Mat ini, int** line, int index){
	Mat res(ini.rows, ini.cols+1, ini.type());
	for(int i = 0; i < res.rows; i++){
		int temp = line[index][i];
		for(int j = 0; j < res.cols; j++){
			if(j < temp){
				res.at<Vec3b>(i, j)[0] = ini.at<Vec3b>(i, j)[0];
				res.at<Vec3b>(i, j)[1] = ini.at<Vec3b>(i, j)[1];
				res.at<Vec3b>(i, j)[2] = ini.at<Vec3b>(i, j)[2];
			}
			else if(j > temp){
				res.at<Vec3b>(i, j)[0] = ini.at<Vec3b>(i, j-1)[0];
				res.at<Vec3b>(i, j)[1] = ini.at<Vec3b>(i, j-1)[1];
				res.at<Vec3b>(i, j)[2] = ini.at<Vec3b>(i, j-1)[2];
			}
			else{
				res.at<Vec3b>(i, j)[0] = (ini.at<Vec3b>(i, j-1)[0]+ini.at<Vec3b>(i, j)[0])/2;
				res.at<Vec3b>(i, j)[1] = (ini.at<Vec3b>(i, j-1)[1]+ini.at<Vec3b>(i, j)[1])/2;
				res.at<Vec3b>(i, j)[2] = (ini.at<Vec3b>(i, j-1)[2]+ini.at<Vec3b>(i, j)[2])/2;
			}
		}
	}
	return res;
}

//图片放大时，显示seam效果（行方向）
Mat row_show(Mat ini, int** line, int index){
	Mat res(ini.rows+1, ini.cols, ini.type());
	for(int i = 0; i < res.cols; i++){
		int temp = line[index][i];
		for(int j = 0; j < res.rows; j++){
			if(j < temp){
				res.at<Vec3b>(j, i)[0] = ini.at<Vec3b>(j, i)[0];
				res.at<Vec3b>(j, i)[1] = ini.at<Vec3b>(j, i)[1];
				res.at<Vec3b>(j, i)[2] = ini.at<Vec3b>(j, i)[2];
			}
			else if(j > temp){
				res.at<Vec3b>(j, i)[0] = ini.at<Vec3b>(j-1, i)[0];
				res.at<Vec3b>(j, i)[1] = ini.at<Vec3b>(j-1, i)[1];
				res.at<Vec3b>(j, i)[2] = ini.at<Vec3b>(j-1, i)[2];
			}
			else{
				res.at<Vec3b>(j, i)[0] = 0;
				res.at<Vec3b>(j, i)[1] = 0;
				res.at<Vec3b>(j, i)[2] = 255;
			}
		}
	}
	return res;
}

//图片放大时，显示seam效果（列方向）
Mat col_show(Mat ini, int** line, int index){
	Mat res(ini.rows, ini.cols+1, ini.type());
	for(int i = 0; i < res.rows; i++){
		int temp = line[index][i];
		for(int j = 0; j < res.cols; j++){
			if(j < temp){
				res.at<Vec3b>(i, j)[0] = ini.at<Vec3b>(i, j)[0];
				res.at<Vec3b>(i, j)[1] = ini.at<Vec3b>(i, j)[1];
				res.at<Vec3b>(i, j)[2] = ini.at<Vec3b>(i, j)[2];
			}
			else if(j > temp){
				res.at<Vec3b>(i, j)[0] = ini.at<Vec3b>(i, j-1)[0];
				res.at<Vec3b>(i, j)[1] = ini.at<Vec3b>(i, j-1)[1];
				res.at<Vec3b>(i, j)[2] = ini.at<Vec3b>(i, j-1)[2];
			}
			else{
				res.at<Vec3b>(i, j)[0] = 0;
				res.at<Vec3b>(i, j)[1] = 0;
				res.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}
	return res;
}

//图片缩小时，显示seam效果（行方向）
Mat add_row_show(Mat ini, int** line, int index){
	Mat res(ini.rows+1, ini.cols, ini.type());
	for(int i = 0; i < res.cols; i++){
		int temp = line[index][i];
		for(int j = 0; j < res.rows; j++){
			if(j < temp){
				res.at<Vec3b>(j, i)[0] = ini.at<Vec3b>(j, i)[0];
				res.at<Vec3b>(j, i)[1] = ini.at<Vec3b>(j, i)[1];
				res.at<Vec3b>(j, i)[2] = ini.at<Vec3b>(j, i)[2];
			}
			else if(j == temp){
				res.at<Vec3b>(j, i)[0] = 0;
				res.at<Vec3b>(j, i)[1] = 0;
				res.at<Vec3b>(j, i)[2] = 255;
			}
			else{
				res.at<Vec3b>(j, i)[0] = ini.at<Vec3b>(j-1, i)[0];
				res.at<Vec3b>(j, i)[1] = ini.at<Vec3b>(j-1, i)[1];
				res.at<Vec3b>(j, i)[2] = ini.at<Vec3b>(j-1, i)[2];
			}
		}
	}
	return res;
}

//图片缩小时，显示seam效果（列方向）
Mat add_col_show(Mat ini, int** line, int index){
	Mat res(ini.rows, ini.cols+1, ini.type());
	for(int i = 0; i < res.rows; i++){
		int temp = line[index][i];
		for(int j = 0; j < res.cols; j++){
			if(j < temp){
				res.at<Vec3b>(i, j)[0] = ini.at<Vec3b>(i, j)[0];
				res.at<Vec3b>(i, j)[1] = ini.at<Vec3b>(i, j)[1];
				res.at<Vec3b>(i, j)[2] = ini.at<Vec3b>(i, j)[2];
			}
			else if(j == temp){
				res.at<Vec3b>(i, j)[0] = 0;
				res.at<Vec3b>(i, j)[1] = 0;
				res.at<Vec3b>(i, j)[2] = 255;
			}
			else{
				res.at<Vec3b>(i, j)[0] = ini.at<Vec3b>(i, j-1)[0];
				res.at<Vec3b>(i, j)[1] = ini.at<Vec3b>(i, j-1)[1];
				res.at<Vec3b>(i, j)[2] = ini.at<Vec3b>(i, j-1)[2];
			}
		}
	}
	return res;
}

//动态规划，寻找能量最小的竖线
//dealt表示是否有保护处理，remove表示是否移除，expand表示是否为图片放大。（“是”时为1）
Mat dp(Mat ini, Mat& deal, int** line, int index, bool dealt, bool remove, bool expand){
	Mat temp;
	cvtColor(ini, temp, CV_BGR2GRAY);                         //转为灰度图
	GaussianBlur(temp, temp, Size(3, 3), 0, 0, BORDER_DEFAULT); //高斯滤波
	Mat gray1(temp.cols, temp.rows, temp.type()), gray2(temp.cols, temp.rows, temp.type()), 
		gray(temp.cols, temp.rows, temp.type());
	Sobel(temp, gray1, temp.depth(), 1, 0, 3);
	Sobel(temp, gray2, temp.depth(), 0, 1, 3);
	add(abs(gray1), abs(gray2), gray);
	if(dealt){
		if(!remove)	gray = protect(deal, gray);
		else gray = remove_object(deal, gray);
	}
	int col = gray.cols;
	int row = gray.rows;
	
	node** M = new node*[col];
	for(int i = 0; i < col; i++){
		M[i] = new node[row];
		for(int j = 0; j < row; j++){
			M[i][j].cont = (int)gray.at<uchar>(j, i);
			M[i][j].pre = -1;
			if(i > 0){
				if(j == 0){
					if(M[i-1][j].cont < M[i-1][j+1].cont){
						M[i][j].cont += M[i-1][j].cont;
						M[i][j].pre = j;
					}
					else{
						M[i][j].cont += M[i-1][j+1].cont;
						M[i][j].pre = j+1;
					}
				}
				else if(j == row-1){
					if(M[i-1][j-1].cont < M[i-1][j].cont){
						M[i][j].cont += M[i-1][j-1].cont;
						M[i][j].pre = j-1;
					}
					else{
						M[i][j].cont += M[i-1][j].cont;
						M[i][j].pre = j;
					}
				}
				else{
					int temp_pre = 0;
					int temp_cont = 0;
					temp_pre = (M[i-1][j-1].cont < M[i-1][j].cont) ? -1 : 0;
					temp_cont = M[i-1][j+temp_pre].cont;
					temp_pre = (temp_cont < M[i-1][j+1].cont) ? temp_pre : 1;
					temp_cont = M[i-1][j+temp_pre].cont;
					M[i][j].cont += temp_cont;
					M[i][j].pre = j+temp_pre;
				}
			}
		}
	}
	int min_energy = -1;
	int min_y = -1;
	for(int i = 0; i < row; i++){
		if(min_energy == -1 || min_energy > M[col-1][i].cont){
			min_energy = M[col-1][i].cont;
			min_y = i;
		}
	}
	for(int i = col-1; i >= 0; i--){
		line[index][i] = min_y;
		min_y = M[i][min_y].pre;
	}
	Mat res = ini;
	if(!expand){
		res = del_row(ini, line, index);
		if(dealt)	deal = del_row(deal, line, index);
	}
	else{
		res = add_row(ini, line, index);
		deal = row_show(deal, line, index);
	}
	for(int i = 0; i < col; i++) delete[] M[i];
	delete[] M;
	return res;
}

//动态规划，寻找能量最小的竖线
//dealt表示是否有保护处理，remove表示是否移除，expand表示是否为图片放大。（“是”时为1）
Mat dp_y(Mat ini, Mat& deal, int** line, int index, bool dealt, bool remove, bool expand){
	Mat temp;
	cvtColor(ini, temp, CV_BGR2GRAY);          //转为灰度图
	GaussianBlur(temp, temp, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat gray1(temp.cols, temp.rows, temp.type()), gray2(temp.cols, temp.rows, temp.type()), 
		gray(temp.cols, temp.rows, temp.type());
	Sobel(temp, gray1, temp.depth(), 1, 0, 3);
	Sobel(temp, gray2, temp.depth(), 0, 1, 3);
	add(abs(gray1), abs(gray2), gray);
	if(dealt){
		if(!remove)	gray = protect(deal, gray);
		else gray = remove_object(deal, gray);
	}
	Mat res = ini;
	int col = gray.cols;
	int row = gray.rows;
	
	node** M = new node*[col];
	for(int i = 0; i < col; i++){
		M[i] = new node[row];
	}
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col; j++){
			M[j][i].cont = (int)gray.at<uchar>(i, j);
			M[j][i].pre = -1;
			if(i > 0){
				if(j == 0){
					if(M[j][i-1].cont < M[j+1][i-1].cont){
						M[j][i].cont += M[j][i-1].cont;
						M[j][i].pre = j;
					}
					else{
						M[j][i].cont += M[j+1][i-1].cont;
						M[j][i].pre = j+1;
					}
				}
				else if(j == col-1){
					if(M[j][i-1].cont < M[j-1][i-1].cont){
						M[j][i].cont += M[j][i-1].cont;
						M[j][i].pre = j;
					}
					else{
						M[j][i].cont += M[j-1][i-1].cont;
						M[j][i].pre = j-1;
					}
				}
				else{
					int temp_pre = 0;
					int temp_cont = 0;
					temp_pre = (M[j-1][i-1].cont < M[j][i-1].cont) ? -1 : 0;
					temp_cont = M[j+temp_pre][i-1].cont;
					temp_pre = (temp_cont < M[j+1][i-1].cont) ? temp_pre : 1;
					temp_cont = M[j+temp_pre][i-1].cont;
					M[j][i].cont += temp_cont;
					M[j][i].pre = j+temp_pre;
				}
			}
		}
	}
	int min_energy = -1;
	int min_x = -1;
	for(int i = 0; i < col; i++){
		if(min_energy == -1 || min_energy > M[i][row-1].cont){
			min_energy = M[i][row-1].cont;
			min_x = i;
		}
	}
	for(int i = row-1; i >= 0; i--){
		line[index][i] = min_x;
		min_x = M[min_x][i].pre;
	}
	if(!expand){
		res = del_col(ini, line, index);
		if(dealt)	deal = del_col(deal, line, index);
	}
	else{
		res = add_col(ini, line, index);
		deal = col_show(deal, line, index);
	}

	for(int i = 0; i < col; i++) delete [] M[i];
	delete[] M;
	return res;
}

//对于每个图片，去掉若干行列
//remove为1表示移除，0表示保护
void work(Mat ini, string filename1, string filename2,  bool remove, string filename3 = ""){
	int** myline_x;
	int** myline_y;
	myline_x = new int*[NUM_X];
	myline_y = new int*[NUM_Y];
	int index_x = 0, index_y = 0;
	for(int i = 0; i < NUM_Y; i++){
		myline_y[i] = new int[max(ini.rows, ini.cols)];
	}
	for(int i = 0; i < NUM_X; i++){
		myline_x[i] = new int[max(ini.rows, ini.cols)];
	}
	Mat res, temp, show_res;
	Mat deal = ini;
	bool dealt = false;
	if(filename3 != ""){
		deal = imread(filename3);
		dealt = true;
	}
	res = dp_y(ini, deal, myline_y, index_y, dealt, remove, false);
	index_y++;
	res = dp(res, deal, myline_x, index_x, dealt, remove, false);
	index_x++;
	while(index_x < ini.rows*0.2 || index_y < ini.cols*0.2){
		if(index_y < ini.cols*0.2){
			res = dp_y(res, deal, myline_y, index_y, dealt, remove, false);
			index_y++;
		}
		if(index_x < ini.rows*0.2){
			res = dp(res, deal, myline_x, index_x, dealt, remove, false);
			index_x++;
		}
	}
	if(index_y > index_x){
		show_res = add_col_show(res, myline_y, index_y);
		while(index_y > index_x){
			index_y--;
			show_res = add_col_show(show_res, myline_y, index_y);
		}
	}
	else{
		show_res = add_row_show(res, myline_x, index_x);
		while(index_x > index_y){
			index_x--;
			show_res = add_row_show(show_res, myline_x, index_x);
		}
	}
	while(index_x > 0){
		index_x--;
		show_res = add_row_show(show_res, myline_x, index_x);
		index_y--;
		show_res = add_col_show(show_res, myline_y, index_y);
	}
	imwrite(filename1, res);
	imwrite(filename2, show_res);
}


//对每个图片，增加200行200列
void work_expand(Mat ini, string filename1, string filename2){
	int** myline_x;
	int** myline_y;
	myline_x = new int*[NUM_X];
	myline_y = new int*[NUM_Y];
	int index_x = 0, index_y = 0;
	for(int i = 0; i < NUM_Y; i++){
		myline_y[i] = new int[max(ini.rows, ini.cols)];
	}
	for(int i = 0; i < NUM_X; i++){
		myline_x[i] = new int[max(ini.rows, ini.cols)];
	}
	Mat res, temp, show_res = ini;
	res = dp_y(ini, show_res, myline_y, index_y, false, false, true);
	index_y++;
	res = dp(res, show_res, myline_x, index_x, false, false, true);
	index_x++;
	while(index_x < 200 || index_y < 200){
		cout << index_x << endl;
		if(index_y < 200){
			res = dp_y(res, show_res, myline_y, index_y, false, false, true);
			index_y++;
		}
		if(index_x < 200){
			res = dp(res, show_res, myline_x, index_x, false, false, true);
			index_x++;
		}
	}
	imwrite(filename1, res);
	imwrite(filename2, show_res);
}

int main(){
	std::string s[6] = {"1.jpg", "2.png", "3.jpg", "4.jpg", "5.jpg", "6.jpg"};
	Mat _image = imread(s[0]);
	work(_image, "show1.bmp", "res1.bmp", false);
	Mat _image1 = imread(s[1]);
	work(_image1, "show2.bmp", "res2.bmp", false);
	Mat _image2 = imread(s[2]);
	work(_image2, "show3.bmp", "res3.bmp", false);
	Mat _image3 = imread(s[3]);
	work(_image3, "show4.bmp", "res4.bmp", false);
	Mat _image4 = imread(s[4]);
	work(_image4, "show5.bmp", "res5.bmp", false);
	Mat _image5 = imread(s[5]);
	work(_image5, "show6.bmp", "res6.bmp", false);
	waitKey(0); 
	return 0;
}