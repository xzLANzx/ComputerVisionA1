#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

void rootSquareDifference(const Mat& mat1, const Mat& mat2, Mat& result){
    result.release();
    result = Mat(mat1.rows, mat1.cols, CV_8UC3);
    for (int i = 0; i < mat1.rows; ++i) {
        for (int j = 0; j < mat1.cols; ++j) {
            Vec3b pixel1 = mat1.at<Vec3b>(i, j);
            Vec3b pixel2 = mat2.at<Vec3b>(i, j);
            unsigned char d1 = sqrt(abs(pow(pixel1[0], 2) - pow(pixel2[0], 2)));
            unsigned char d2 = sqrt(abs(pow(pixel1[1], 2) - pow(pixel2[1], 2)));
            unsigned char d3 = sqrt(abs(pow(pixel1[2], 2) - pow(pixel2[2], 2)));
            result.at<Vec3b>(i, j) = Vec3b(d1, d2, d3);
        }
    }
}

int main() {
    cout<<"Please choose the image you want to see: "<<endl;
    cout<<"1. old well"<<endl;
    cout<<"2. crayons"<<endl;
    cout<<"3. pencils"<<endl;
    int option;
    cin>>option;

    while((option != 1) && (option != 2) && (option !=3) ){
        cout<<"Wrong option! Please enter 1, 2 or 3"<<endl;
        cin>>option;
    }

    //read data into img
    Mat img1 = imread("image_set/oldwell_mosaic.bmp", IMREAD_COLOR);
    Mat img2 = imread("image_set/crayons_mosaic.bmp", IMREAD_COLOR);
    Mat img3 = imread("image_set/pencils_mosaic.bmp", IMREAD_COLOR);

    Mat img_orig1 = imread("image_set/oldwell.jpg", IMREAD_COLOR);
    Mat img_orig2 = imread("image_set/crayons.jpg", IMREAD_COLOR);
    Mat img_orig3 = imread("image_set/pencils.jpg", IMREAD_COLOR);

    Mat img, img_orig;
    if(option == 1){
        img = img1;
        img_orig = img_orig1;
    }else if(option == 2){
        img = img2;
        img_orig = img_orig2;
    }else if(option == 3){
        img = img3;
        img_orig = img_orig3;
    }


    if (!img.data || !img_orig.data) {
        cout << "Image not loaded" << endl;
        system("pause");
        exit(-1);
    }


    //split img into 3 channels
    Mat channels[3];
    split(img, channels);


    //blue chess board
    for (int i = 0; i < channels[0].rows; ++i)
        if (i % 2 == 1) channels[0].row(i).setTo(Scalar(0));
    for (int j = 0; j < channels[0].cols; ++j)
        if (j % 2 == 1) channels[0].col(j).setTo(Scalar(0));

    //green chess board
    for (int i = 0; i < channels[1].rows; ++i)
        if (i % 2 == 0) channels[1].row(i).setTo(Scalar(0));
    for (int j = 0; j < channels[1].cols; ++j)
        if (j % 2 == 0) channels[1].col(j).setTo(Scalar(0));

    //red chess board
    for (int i = 0; i < channels[2].rows; ++i) {
        for (int j = 0; j < channels[2].cols; ++j) {
            if (i % 2 == 0) {
                if (j % 2 == 0) channels[2].at<uchar>(i, j) = 0;
            } else {
                if (j % 2 == 1) channels[2].at<uchar>(i, j) = 0;
            }
        }
    }



    //create 3 filters
    float kb_data[9] = {1.0f / 4, 1.0f / 2, 1.0f / 4,
                        1.0f / 2, 1.0f / 1, 1.0f / 2,
                        1.0f / 4, 1.0f / 2, 1.0f / 4};

    float kg_data[9] = {1.0f / 4, 1.0f / 2, 1.0f / 4,
                        1.0f / 2, 1.0f / 1, 1.0f / 2,
                        1.0f / 4, 1.0f / 2, 1.0f / 4};

    float kr_data[9] = {0.0f, 1.0f / 4, 0.0f,
                        1.0f / 4, 1.0f, 1.0f / 4,
                        0.0f, 1.0f / 4, 0.0f};


    Mat kernel_b = Mat(3, 3, CV_32F, kb_data);
    Mat kernel_g = Mat(3, 3, CV_32F, kg_data);
    Mat kernel_r = Mat(3, 3, CV_32F, kr_data);

    //interpolate 3 channels separately
    Mat img_dst_b, img_dst_g, img_dst_r;
    filter2D(channels[0], img_dst_b, channels[0].depth(), kernel_b);
    filter2D(channels[1], img_dst_g, channels[1].depth(), kernel_g);
    filter2D(channels[2], img_dst_r, channels[2].depth(), kernel_r);


    vector<Mat> rbgImage;
    rbgImage.push_back(img_dst_b);
    rbgImage.push_back(img_dst_g);
    rbgImage.push_back(img_dst_r);
    Mat image;
    merge(rbgImage, image);

    //compute part 1 root square
    Mat img_sqrt;
    rootSquareDifference(img_orig, image, img_sqrt);

    //cout<<sum(img_sqrt)<<endl;

    Mat comparison;
    hconcat(img_orig,image,comparison);
    hconcat(comparison,img_sqrt,comparison);
    imshow("Part 1", comparison);


    Mat G_R, B_R, G_R_Blur, B_R_Blur;
    img_dst_b.convertTo(img_dst_b, CV_32F);
    img_dst_r.convertTo(img_dst_r, CV_32F);
    img_dst_g.convertTo(img_dst_g, CV_32F);

    cv::subtract(img_dst_g,img_dst_r,G_R);
    cv::subtract(img_dst_b,img_dst_r,B_R);

    //median filtering
    medianBlur(G_R, G_R_Blur, 5);
    medianBlur(B_R, B_R_Blur, 5);

    //add back
    img_dst_g.release();
    img_dst_b.release();
    cv::add(img_dst_r, G_R_Blur, img_dst_g);
    cv::add(img_dst_r, B_R_Blur, img_dst_b);

    //reconstruct image
    img_dst_b.convertTo(img_dst_b, CV_8U);
    img_dst_r.convertTo(img_dst_r, CV_8U);
    img_dst_g.convertTo(img_dst_g, CV_8U);

    rbgImage.clear();
    rbgImage.push_back(img_dst_b);
    rbgImage.push_back(img_dst_g);
    rbgImage.push_back(img_dst_r);

    image.release();
    merge(rbgImage, image);

    //compute part 2root square
    rootSquareDifference(img_orig, image, img_sqrt);
    //cout<<sum(img_sqrt)<<endl;

    comparison.release();
    hconcat(img_orig,image,comparison);
    hconcat(comparison,img_sqrt,comparison);
    imshow("Part 2", comparison);

    waitKey(0);

    return 0;
}