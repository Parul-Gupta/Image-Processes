#include <stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include<cmath>
#include<vector>
using namespace cv;
using namespace std;

void negative(Mat& img);
void logtrans(Mat& img);
void gammatrans(Mat& img);
void bitslice(Mat& img);
void rotate(Mat& img);
void translate(Mat& img);
void scale(Mat& img);
void shear(Mat& img);
void hist_eq(Mat& img);
void hist_match(Mat& img);
void adaptive(Mat& img);
void con_stretch(Mat& img);
void gray_slice(Mat& img);
void tie(Mat& img);

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], IMREAD_GRAYSCALE );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    int f=0;
    cout<<"Enter:\n1 to get input image negative\n2 for its log transformation\n3 for power transformation\n4 for bitplane slicing\n5 for rotating\n6 for translation\n7 for scaling(resizing)\n8 for shearing\n9 for histogram equalisation\n10 for histogram matching\n11 for adaptive histogram equalisation\n12 for contrast stretching\n13 for gray level slicing and\n14 for reconstruction using tie points: ";
    cin>>f;
    if(f==1){negative(image);}
    else if(f==2){logtrans(image);}
    else if(f==3){gammatrans(image);}
    else if(f==4){bitslice(image);}
    else if(f==5){rotate(image);}
    else if(f==6){translate(image);}
    else if(f==7){scale(image);}
    else if(f==8){shear(image);}
    else if(f==9){hist_eq(image);}
    else if(f==10){hist_match(image);}
    else if(f==11){adaptive(image);}
    else if(f==12){con_stretch(image);}
    else if(f==13){gray_slice(image);}
    else if(f==14){tie(image);}
    
    return 0;
}

void negative(Mat& img){
	int x=img.rows,y=img.cols;
	Mat nimg = Mat::zeros(x,y,CV_8UC1);
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			nimg.at<uchar>(i,j)=255-img.at<uchar>(i,j);
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Negative Image", WINDOW_AUTOSIZE );
    imshow("Negative Image", nimg);
    waitKey(0);
    return;
}

void logtrans(Mat& img){
	int x=img.rows,y=img.cols;
	Mat logimg = Mat::zeros(x,y,CV_8UC1);
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			logimg.at<uchar>(i,j)=20*log(1+img.at<uchar>(i,j));
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Logarithm Image", WINDOW_AUTOSIZE );
    imshow("Logarithm Image", logimg);
	waitKey(0);
    return;
}

void gammatrans(Mat& img){
	int x=img.rows,y=img.cols;
	float gamma=0;
	cout<<"Enter the value of gamma: ";
	cin>>gamma;
	cout<<endl;
	Mat gammaimg = Mat::zeros(x,y,CV_8UC1);
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			gammaimg.at<uchar>(i,j)=pow(img.at<uchar>(i,j),gamma);
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Power Image", WINDOW_AUTOSIZE );
    imshow("Power Image", gammaimg);
	waitKey(0);
    return;
}

void bitslice(Mat& img){
	int x=img.rows,y=img.cols,n=0;
	cout<<"Enter the bit-plane number(1-8) for slicing: ";
	cin>>n;
	cout<<endl;
	Mat bitimg = Mat::zeros(x,y,CV_8UC1);
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			bitimg.at<uchar>(i,j)=(int)img.at<uchar>(i,j) & (int)pow(2,n-1);
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Sliced Image", WINDOW_AUTOSIZE );
    imshow("Sliced Image", bitimg);
	waitKey(0);
    return;
}

void con_stretch(Mat& img){
	int x=img.rows,y=img.cols;
	int r1=0,s1=0,r2=0,s2=0;
	cout<<"Enter the first input intensity level and corresponding output level (i.e. r1 s1): ";
	cin>>r1>>s1;
	cout<<"Enter the second input intensity level and corresponding output level (i.e. r2 s2): ";
	cin>>r2>>s2;
	if(r1>r2){cout<<"Wrong order of inputs : r1<r2"<<endl;return;}
	if(r1>255 || r2>255 || s1>255 || s2>255){cout<<"Wrong intensity values (should be <256)."<<endl;return;}
	Mat stretchimg = Mat::zeros(x,y,CV_8UC1);
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			if(img.at<uchar>(i,j)<=r1){
				stretchimg.at<uchar>(i,j)=(int)(img.at<uchar>(i,j)*(float)s1/(float)r1);
			}
			else if(img.at<uchar>(i,j)<=r2){
				stretchimg.at<uchar>(i,j)=s1+(int)((img.at<uchar>(i,j)-r1)*(float)(s2-s1)/(float)(r2-r1));
			}
			else stretchimg.at<uchar>(i,j)=s2+(int)((img.at<uchar>(i,j)-r2)*(float)(255-s2)/(float)(255-r2));
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Stretched Image", WINDOW_AUTOSIZE );
    imshow("Stretched Image", stretchimg);
	waitKey(0);
    return;
}

void gray_slice(Mat& img){
	int x=img.rows,y=img.cols;
	int s1=0,s2=0;
	Mat sliceimg=Mat::zeros(x,y,CV_8UC1);
	cout<<"Enter the range of gray levels which you wish to highlight (start end): ";
	cin>>s1>>s2;
	if(s1>255 || s2>255){cout<<"Wrong intensity levels.(should be < 256)\n";return;}
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			if(img.at<uchar>(i,j)>=s1 && img.at<uchar>(i,j)<=s2){
				sliceimg.at<uchar>(i,j)=255;
			}
			else sliceimg.at<uchar>(i,j)=img.at<uchar>(i,j);
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Sliced Image", WINDOW_AUTOSIZE );
    imshow("Sliced Image", sliceimg);
	waitKey(0);
    return;
}

void rotate(Mat& img){
	float n=0;
	cout<<"Enter the angle (in degrees) by which you wish to rotate the image anti-clockwise: ";
	cin>>n;
	if(n<0)n=360+n;
	n=n*0.0174533;
	char c='b';
	cout<<"Enter 1 for nearest neighbourhood interpolation and any other key for bilinear interpolation: ";
	cin>>c;
	int x=img.rows,y=img.cols;
	Mat rotimg = Mat::zeros(abs(x*cos(n))+abs(y*sin(n)),abs(y*cos(n))+abs(x*sin(n)),CV_8UC1);
	float cx=rotimg.rows/2;
	float cy=rotimg.cols/2;
	float dx=cx*cos(n)+cy*sin(n);
	float dy=cy*cos(n)-cx*sin(n);
	dx=abs(dx-x/2.0);
	dy=abs(dy-y/2.0);
	for(int i=0;i<rotimg.rows;i++){
		for(int j=0;j<rotimg.cols;j++){
			float px;
			if(n<=1.5707)px=i*cos(n)+j*sin(n)-dx;
			else px=i*cos(n)+j*sin(n)+dx;
			float py;
			if(n<=4.7123)py=j*cos(n)-i*sin(n)+dy;
			else py=j*cos(n)-i*sin(n)-dy;
			if((int)px<0 || (int)px>x || (int)py<0 || (int)py>y)rotimg.at<uchar>(i,j)=0;
			else{
				float rx=px-(int)px;
				float ry=py-(int)py;
				if(c=='1'){
					if(rx<0.5 && ry<0.5)rotimg.at<uchar>(i,j)=img.at<uchar>((int)px,(int)py);
					else if(rx<0.5)rotimg.at<uchar>(i,j)=img.at<uchar>((int)px,(int)py+1);
					else if(ry<0.5)rotimg.at<uchar>(i,j)=img.at<uchar>((int)px+1,(int)py);
					else rotimg.at<uchar>(i,j)=img.at<uchar>((int)px+1,(int)py+1);
				}
				else rotimg.at<uchar>(i,j)=(1-rx)*(1-ry)*img.at<uchar>((int)px,(int)py)+(1-rx)*ry*img.at<uchar>((int)px,(int)py+1)+(1-ry)*rx*img.at<uchar>((int)px+1,(int)py)+rx*ry*img.at<uchar>((int)px+1,(int)py+1);
			}
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Rotated Image", WINDOW_AUTOSIZE );
    imshow("Rotated Image", rotimg);
	waitKey(0);
    return;
}

void translate(Mat& img){
	int X=0,Y=0;
	cout<<"Enter the distance by which you wish to move the image horizontally: ";
	cin>>Y;
	cout<<"Enter the distance by which you wish to move the image vertically: ";
	cin>>X;
	int x=img.rows,y=img.cols;
	Mat transimg=Mat::zeros(x,y,CV_8UC1);
	x=min(x,X+x);
	y=min(y,Y+y);
	for(int i=max(0,X);i<x;i++){
		for(int j=max(0,Y);j<y;j++){
			transimg.at<uchar>(i,j)=img.at<uchar>(i-X,j-Y);
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Translated Image", WINDOW_AUTOSIZE );
    imshow("Translated Image", transimg);
	waitKey(0);
    return;
}

void scale(Mat& img){
	float SX=1,SY=1;
	cout<<"Enter the factor by which you wish to scale the image horizontally: ";
	cin>>SY;
	cout<<"Enter the factor by which you wish to scale the image vertically: ";
	cin>>SX;
	int x=img.rows,y=img.cols;
	Mat scaleimg=Mat::zeros(SX*x,SY*y,CV_8UC1);
	char c='a';
	cout<<"Enter 1 for nearest neighbourhood interpolation and any other key for bilinear interpolation: ";
	cin>>c;
	if(c!='1'){
		for(float i=0;i<SX*x;i++){
			for(float j=0;j<SY*y;j++){
				float px=i/SX,py=j/SY;
				float rx=px-(int)px;
				float ry=py-(int)py;
				scaleimg.at<uchar>(i,j)=(1-rx)*(1-ry)*img.at<uchar>((int)px,(int)py)+rx*(1-ry)*img.at<uchar>((int)px+1,(int)py)+ry*(1-rx)*img.at<uchar>((int)px,(int)py+1)+ry*rx*img.at<uchar>((int)px+1,(int)py+1);
			}
		}
	}
	else{
		for(float i=0;i<SX*x;i++){
			for(float j=0;j<SY*y;j++){
				float px=i/SX,py=j/SY;
				float rx=px-(int)px;
				float ry=py-(int)py;
				if(rx<0.5)px=(int)px;
				else px=(int)px+1;
				if(ry<0.5)py=(int)py;
				else py=(int)py+1;
				scaleimg.at<uchar>(i,j)=img.at<uchar>(px,py);
			}
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", img);

	namedWindow("Scaled Image", WINDOW_AUTOSIZE );
    imshow("Scaled Image", scaleimg);
	waitKey(0);
    return;
}

void shear(Mat& img){
	int x=img.rows,y=img.cols;
	char s='h';
	cout<<"Enter h if you want to shear the image horizontally and any other key if you want to shear the image vertically: ";
	cin>>s;
	float f=1;
	cout<<"Enter the factor by which you wish to shear: ";
	cin>>f;
	char c='a';
	cout<<"Enter 1 for nearest neighbourhood interpolation and any other key for bilinear interpolation: ";
	cin>>c;
	Mat shearimg;
	if(s=='h'){
		shearimg=Mat::zeros(x,y+abs(f)*x,CV_8UC1);
		for(float i=0;i<x;i++){
			if(f>=0){
				for(float j=f*i;j<y+f*i;j++){
					float py=j-f*i;
					float ry=py-(int)py;
					if(c=='1'){
						if(ry>=0.5)shearimg.at<uchar>(i,j)=img.at<uchar>(i,(int)py+1);
						else shearimg.at<uchar>(i,j)=img.at<uchar>(i,(int)py);
					}
					else shearimg.at<uchar>(i,j)=(1-ry)*img.at<uchar>(i,(int)py)+ry*img.at<uchar>(i,(int)py+1);
				}
			}
			else{
				for(float j=f*(i-x);j<y+f*(i-x);j++){
					float py=j-f*(i-x);
					float ry=py-(int)py;
					if(c=='1'){
						if(ry>=0.5)shearimg.at<uchar>(i,j)=img.at<uchar>(i,(int)py+1);
						else shearimg.at<uchar>(i,j)=img.at<uchar>(i,(int)py);
					}
					else shearimg.at<uchar>(i,j)=(1-ry)*img.at<uchar>(i,(int)py)+ry*img.at<uchar>(i,(int)py+1);
				}
			}
		}
	}
	else{
		shearimg=Mat::zeros(x+abs(f)*y,y,CV_8UC1);
		for(float j=0;j<y;j++){
			if(f>=0){
				for(float i=f*j;i<x+f*j;i++){
					float px=i-f*j;
					float rx=px-(int)px;
					if(c=='1'){
						if(rx>=0.5)shearimg.at<uchar>(i,j)=img.at<uchar>((int)px+1,j);
						else shearimg.at<uchar>(i,j)=img.at<uchar>((int)px,j);
					}
					else shearimg.at<uchar>(i,j)=(1-rx)*img.at<uchar>((int)px,j)+rx*img.at<uchar>((int)px+1,j);
				}
			}
			else{
				for(float i=f*(j-y);i<x+f*(j-y);i++){
					float px=i-f*(j-y);
					float rx=px-(int)px;
					if(c=='1'){
						if(rx>=0.5)shearimg.at<uchar>(i,j)=img.at<uchar>((int)px+1,j);
						else shearimg.at<uchar>(i,j)=img.at<uchar>((int)px,j);
					}
					else shearimg.at<uchar>(i,j)=(1-rx)*img.at<uchar>((int)px,j)+rx*img.at<uchar>((int)px+1,j);
				}
			}
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", img);

	namedWindow("Sheared Image", WINDOW_AUTOSIZE );
	imshow("Sheared Image", shearimg);	
	waitKey(0);
    return;
}

void hist_eq(Mat& img){
	int x=img.rows,y=img.cols;
	Mat hist_eqimg=Mat::zeros(x,y,CV_8UC1);
	float hist[256]={0};
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			hist[img.at<uchar>(i,j)]++;
		}
	}
	for(int i=0;i<256;i++){
		hist[i]=hist[i]/(x*y);
	}
	float cum_hist[256]={0};
	cum_hist[0]=hist[0];
	for(int i=1;i<256;i++){
		cum_hist[i]=hist[i]+cum_hist[i-1];
	}
	for(int i=0;i<256;i++){
		cum_hist[i]=255*cum_hist[i];
	}
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			hist_eqimg.at<uchar>(i,j)=(int)cum_hist[img.at<uchar>(i,j)];
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", img);

	namedWindow("Histogram Equalised Image", WINDOW_AUTOSIZE );
	imshow("Histogram Equalised Image", hist_eqimg);	
	waitKey(0);
    return;
}

void hist_match(Mat& img){
	Mat match;
	string s="";
	cout<<"Enter the path of the other image to be equalised: ";
	cin>>s;
    match = imread( s, IMREAD_GRAYSCALE );
    if ( !match.data )
    {
        printf("No image data \n");
        return;
    }
    int x=img.rows,y=img.cols;
    int x1=match.rows,y1=match.cols;
    float hist_img[256]={0};
    float hist_match[256]={0};
    for(int i=0;i<x;i++){
    	for(int j=0;j<y;j++){
    		hist_img[img.at<uchar>(i,j)]++;
    	}
    }
    for(int i=0;i<x1;i++){
    	for(int j=0;j<y1;j++){
    		hist_match[match.at<uchar>(i,j)]++;
    	}
    }
    for(int i=0;i<256;i++){
    	hist_img[i]=hist_img[i]/(x*y);
    	hist_match[i]=hist_match[i]/(x1*y1);
    }
    for(int i=1;i<256;i++){
    	hist_img[i]=hist_img[i-1]+hist_img[i];
    	hist_match[i]=hist_match[i-1]+hist_match[i];
    }
    for(int i=0;i<256;i++){
    	hist_img[i]=255*hist_img[i];
    	hist_match[i]=255*hist_match[i];
    }
    vector<int> v[256];
    for(int i=0;i<256;i++){
    	v[(int)hist_match[i]].push_back(i);
    }
    Mat hist_matchimg=Mat::zeros(x,y,CV_8UC1);
    for(int i=0;i<x;i++){
    	for(int j=0;j<y;j++){
    		int size=v[(int)hist_img[img.at<uchar>(i,j)]].size();
    		if(size==1){
    			hist_matchimg.at<uchar>(i,j)=v[(int)hist_img[img.at<uchar>(i,j)]][0];
    		}
    		else if(size>1){
    			int diff=abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]][0]);
    			int ind=0;
    			for(int k=1;k<size;k++){
    				if(diff>abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]][k])){
    					diff=abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]][k]);
    					ind=k;
    				}
    			}
    			hist_matchimg.at<uchar>(i,j)=v[(int)hist_img[img.at<uchar>(i,j)]][ind];
    		}
    		else{
    			int m=1;
    			bool b=true;
    			while(1){
    				if((int)hist_img[img.at<uchar>(i,j)]>0){if(v[(int)hist_img[img.at<uchar>(i,j)]-m].size()>0)break;}
    				else if(v[(int)hist_img[img.at<uchar>(i,j)]+m].size()>0){b=false;break;}
    				m++;
    			}
    			if(b){
					if(v[(int)hist_img[img.at<uchar>(i,j)]-m].size()==1){
						hist_matchimg.at<uchar>(i,j)=v[(int)hist_img[img.at<uchar>(i,j)]-m][0];
					}
					else{
						int diff1=abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]-m][0]);
						int ind1=0;
						for(int k=1;k<v[(int)hist_img[img.at<uchar>(i,j)]-m].size();k++){
							if(diff1>abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]-m][k])){
								diff1=abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]-m][k]);
								ind1=k;
							}
						}
						hist_matchimg.at<uchar>(i,j)=v[(int)hist_img[img.at<uchar>(i,j)]-m][ind1];
					}
				}
				else{
					if(v[(int)hist_img[img.at<uchar>(i,j)]+m].size()==1){
						hist_matchimg.at<uchar>(i,j)=v[(int)hist_img[img.at<uchar>(i,j)]+m][0];
					}
					else{
						int diff1=abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]+m][0]);
						int ind1=0;
						for(int k=1;k<v[(int)hist_img[img.at<uchar>(i,j)]+m].size();k++){
							if(diff1>abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]+m][k])){
								diff1=abs(img.at<uchar>(i,j)-v[(int)hist_img[img.at<uchar>(i,j)]+m][k]);
								ind1=k;
							}
						}
						hist_matchimg.at<uchar>(i,j)=v[(int)hist_img[img.at<uchar>(i,j)]+m][ind1];
					}
				}
    		}
    	}
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", img);
	
	namedWindow("Match Image", WINDOW_AUTOSIZE );
	imshow("Match Image", match);

	namedWindow("Histogram Matched Image", WINDOW_AUTOSIZE );
	imshow("Histogram Matched Image", hist_matchimg);	
	waitKey(0);
    return;
}

void adaptive(Mat& img){
	int x=img.rows,y=img.cols;
	int X=0,Y=0;
	cout<<"Enter the window size(#rows #columns): ";
	cin>>X>>Y;
	if(X>x || Y>y){cout<<"Window size can't be greater than the image size.\n";return;}
	Mat adaimg=Mat::zeros(x,y,CV_8UC1);
	for(int i=0;i<x;i++){
		for(int j=0;j<y;j++){
			float hist[256]={0};
			for(int I=i-X/2;I<i-X/2+X;I++){
				for(int J=j-Y/2;J<j-Y/2+Y;J++){
					if(I>x && J<y) hist[img.at<uchar>(2*x-I,abs(J))]++;
					else if(I>x && J>y) hist[img.at<uchar>(2*x-I,2*y-J)]++;
					else if(I<x && J>y) hist[img.at<uchar>(abs(I),2*y-J)]++;
					else hist[img.at<uchar>(abs(I),abs(J))]++;
				}
			}
			for(int I=0;I<256;I++){
				hist[I]=hist[I]/(X*Y);
			}
			for(int I=1;I<256;I++){
				hist[I]=hist[I]+hist[I-1];
			}
			for(int I=0;I<256;I++){
				hist[I]=255*hist[I];
			}
			adaimg.at<uchar>(i,j)=(int)hist[img.at<uchar>(i,j)];	
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", img);

	namedWindow("Adaptive Image", WINDOW_AUTOSIZE );
	imshow("Adaptive Image", adaimg);	
	waitKey(0);
    return;
}

void tie(Mat& img){
	int x1=0,y1=0,x2=0,y2=0,x3=0,y3=0,x4=0,y4=0;
	cout<<"Enter the original(undistorted) 4 pairs of points (i.e. x1 y1 x2 y2 x3 y3 x4 y4): ";
	cin>>x1>>y1>>x2>>y2>>x3>>y3>>x4>>y4;
	int X1=0,Y1=0,X2=0,Y2=0,X3=0,Y3=0,X4=0,Y4=0;
	cout<<"Enter the corres distorted pairs of points (i.e. x1' y1' x2' y2' x3' y3' x4' y4'): ";
	cin>>X1>>Y1>>X2>>Y2>>X3>>Y3>>X4>>Y4;
	Mat m=Mat::zeros(8,8,CV_32F);
	Mat col=Mat::zeros(8,1,CV_32F);
	m.at<float>(0,0)=x1;
	m.at<float>(0,1)=y1;
	m.at<float>(0,2)=x1*y1;
	m.at<float>(0,3)=1;
	m.at<float>(1,4)=x1;
	m.at<float>(1,5)=y1;
	m.at<float>(1,6)=x1*y1;
	m.at<float>(1,7)=1;
	m.at<float>(2,0)=x2;
	m.at<float>(2,1)=y2;
	m.at<float>(2,2)=x2*y2;
	m.at<float>(2,3)=1;
	m.at<float>(3,4)=x2;
	m.at<float>(3,5)=y2;
	m.at<float>(3,6)=x2*y2;
	m.at<float>(3,7)=1;
	m.at<float>(4,0)=x3;
	m.at<float>(4,1)=y3;
	m.at<float>(4,2)=x3*y3;
	m.at<float>(4,3)=1;
	m.at<float>(5,4)=x3;
	m.at<float>(5,5)=y3;
	m.at<float>(5,6)=x3*y3;
	m.at<float>(5,7)=1;
	m.at<float>(6,0)=x4;
	m.at<float>(6,1)=y4;
	m.at<float>(6,2)=x4*y4;
	m.at<float>(6,3)=1;
	m.at<float>(7,4)=x4;
	m.at<float>(7,5)=y4;
	m.at<float>(7,6)=x4*y4;
	m.at<float>(7,7)=1;
	col.at<float>(0,0)=X1;
	col.at<float>(1,0)=Y1;
	col.at<float>(2,0)=X2;
	col.at<float>(3,0)=Y2;
	col.at<float>(4,0)=X3;
	col.at<float>(5,0)=Y3;
	col.at<float>(6,0)=X4;
	col.at<float>(7,0)=Y4;
	Mat res=Mat::zeros(8,1,CV_32F);
	res=m.inv()*col;
	int X=0,Y=0;
	cout<<"Enter #rows #cols of the original image: ";
	cin>>X>>Y;
	Mat tieimg=Mat::zeros(X,Y,CV_8UC1);
	char c='b';
	cout<<"Enter 1 for nearest neighbour interpolation and any other key for bilinear interpolation: ";
	cin>>c;
	for(int i=0;i<X;i++){
		for(int j=0;j<Y;j++){
			float px=res.at<float>(0,0)*i+res.at<float>(1,0)*j+res.at<float>(2,0)*i*j+res.at<float>(3,0);
			float py=res.at<float>(4,0)*i+res.at<float>(5,0)*j+res.at<float>(6,0)*i*j+res.at<float>(7,0);
			float rx=px-(int)px;
			float ry=py-(int)py;
			if(c=='1'){
				if(rx<0.5 && ry<0.5)tieimg.at<uchar>(i,j)=img.at<uchar>((int)px,(int)py);
				else if(rx<0.5)tieimg.at<uchar>(i,j)=img.at<uchar>((int)px,(int)py+1);
				else if(ry<0.5)tieimg.at<uchar>(i,j)=img.at<uchar>((int)px+1,(int)py);
				else tieimg.at<uchar>(i,j)=img.at<uchar>((int)px+1,(int)py+1);
			}
			else tieimg.at<uchar>(i,j)=(1-rx)*(1-ry)*img.at<uchar>((int)px,(int)py)+rx*(1-ry)*img.at<uchar>((int)px+1,(int)py)+ry*(1-rx)*img.at<uchar>((int)px,(int)py+1)+ry*rx*img.at<uchar>((int)px+1,(int)py+1);
		}
	}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", img);
	
	namedWindow("Reconstructed Image", WINDOW_AUTOSIZE );
	imshow("Reconstructed Image", tieimg);	
	waitKey(0);
    return;
}

















