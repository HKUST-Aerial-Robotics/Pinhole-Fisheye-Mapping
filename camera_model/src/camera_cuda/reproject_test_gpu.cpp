#include <cv_bridge/cv_bridge.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <camera_model/camera_cuda/PolyFisheyeCameraCUDA.h>
#include <code_utils/eigen_utils.h>
#include <code_utils/math_utils/Polynomial.h>


ros::Publisher pub_point;

std::string camera_model_file =
//    "/home/gao/ws2/devel/lib/camera_model/blue4265_fisheye_camera_calib.yaml";
    "/home/gao/ws2/src/Camera/fisheye_model/config/blue621_camera_calib.yaml";
std::string image_name =
//    "/home/gao/ws2/devel/lib/camera_model/blue4265/IMG00000000000000000613.png";
    "/home/gao/ws2/src/Camera/fisheye_model/image/src.png";
int
main(int argc, char** argv)
{
    ros::init(argc, argv, "reproject_test");
    ros::NodeHandle n("~");

    pub_point = n.advertise<sensor_msgs::PointCloud2>("Pont", 100);

    n.getParam("camera_model", camera_model_file);
    n.getParam("image_name", image_name);

    std::cout << "#INFO: camera config is " << camera_model_file << std::endl;
    camera_model::PolyFisheyeCUDA* cam;
    cam = new camera_model::PolyFisheyeCUDA(camera_model_file);

    std::cout << "#INFO: image is " << image_name << std::endl;
    cv::Mat image_in = cv::imread(image_name);

    int w, h, level;
    level = 0;
    w     = (int) (image_in.cols);
    h     = (int) (image_in.rows);

    sensor_msgs::PointCloud2 imagePoint;
    imagePoint.header.stamp    = ros::Time::now();
    imagePoint.header.frame_id = "world";
    imagePoint.height          = h;
    imagePoint.width           = w;
    imagePoint.fields.resize(4);
    imagePoint.fields[0].name     = "x";
    imagePoint.fields[0].offset   = 0;
    imagePoint.fields[0].count    = 1;
    imagePoint.fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    imagePoint.fields[1].name     = "y";
    imagePoint.fields[1].offset   = 4;
    imagePoint.fields[1].count    = 1;
    imagePoint.fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    imagePoint.fields[2].name     = "z";
    imagePoint.fields[2].offset   = 8;
    imagePoint.fields[2].count    = 1;
    imagePoint.fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    imagePoint.fields[3].name     = "rgb";
    imagePoint.fields[3].offset   = 12;
    imagePoint.fields[3].count    = 1;
    imagePoint.fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    imagePoint.is_bigendian       = false;
    imagePoint.point_step         = sizeof(float) * 4;
    imagePoint.row_step           = imagePoint.point_step * imagePoint.width;
    imagePoint.data.resize(imagePoint.row_step * imagePoint.height);
    imagePoint.is_dense = true;

    double theta_max = 0;
    int    i         = 0;
    double t0        = cv::getTickCount();

    int error_num = 0;
    int num = 0;
    for (int row_index = 0; row_index < image_in.rows; ++row_index)
    {
        for (int col_index = 0; col_index < image_in.cols; ++col_index, ++i)
        {
            Eigen::Vector2d p_u(col_index, row_index);
            Eigen::Vector3d P;

            Eigen::Vector2d p_u2;
            cam->liftProjective(p_u, P);
            cam->spaceToPlane(P, p_u2);

            double theta = acos(P(2) / P.norm());
            double phi   = atan2(P(1), P(0));

            if (theta > theta_max)
                theta_max = theta;

            if( col_index == (int)(p_u2(0))
                 && row_index == (int)(p_u2(1)) )
            {
//              std::cout << P.transpose();
//              std::cout << "\033[32;40;1m" << (p_u-p_u2).transpose()
//                        << "\033[0m" << std::endl;
            }
            else
            {
//              std::cout << P.transpose();
//                std::cout << "\033[32;40;1m" << (p_u-p_u2).transpose()
//                          << "\033[0m" << std::endl;
              ++error_num;
            }
            ++num;

            P.normalize();
            float x = P(0) * 1.0;
            float y = P(1) * 1.0;
            float z = P(2) * 1.0;

            int32_t rgb;
            if (//(abs(x) < 0.1 && abs(y) < 0.1) ||
                ((theta) < 90.1 / 57.29 && (theta) > 89.9 / 57.29) ||
                ((theta) < 110.1 / 57.29 && (theta) > 109.9 / 57.29) ||
                ((theta) < 120.1 / 57.29 && (theta) > 119.9 / 57.29) //||
               // ((theta) < 70.1 / 57.29 && (theta) > 69.9 / 57.29) ||
               // (abs(phi) < 135.1 / 57.29 && abs(phi) > 134.9 / 57.29) ||
               // (abs(phi) < 45.1 / 57.29 && abs(phi) > 44.9 / 57.29) ||
               // (abs(phi) > 179.9 / 57.29)
                )
            {
                rgb = ((uchar) 0 << 16) | ((uchar) 255 << 8) | (uchar) 0;
            }
//            else if ((abs(phi) < 0.1 / 57.29))
//            {
//                rgb = ((uchar) 0 << 16) | ((uchar) 255 << 8) | (uchar) 0;
//            }
//            else if ((abs(phi) < 90.1 / 57.29 && abs(phi) > 89.9 / 57.29))
//            {
//                rgb = ((uchar) 0 << 16) | ((uchar) 0 << 8) | (uchar) 255;
//            }
            else
            {
//                uint g = (uchar) image_in.at<uchar>(row_index, col_index);
//                rgb = (g << 16) | (g << 8) | g;

                uint gr = (uchar) image_in.at<cv::Vec3b>(row_index, col_index)[0];
                uint gg = (uchar) image_in.at<cv::Vec3b>(row_index, col_index)[1];
                uint gb = (uchar) image_in.at<cv::Vec3b>(row_index, col_index)[2];
                rgb = (gb << 16) | (gg << 8) | gr;
            }

            memcpy(&imagePoint.data[i * imagePoint.point_step + 0], &x,
                   sizeof(float));
            memcpy(&imagePoint.data[i * imagePoint.point_step + 4], &y,
                   sizeof(float));
            memcpy(&imagePoint.data[i * imagePoint.point_step + 8], &z,
                   sizeof(float));
            memcpy(&imagePoint.data[i * imagePoint.point_step + 12], &rgb,
                   sizeof(int32_t));
        }
    }
    std::cout << "error_num "<<error_num<<std::endl;
    std::cout << "total_num "<<num<<std::endl;

    double t1 = cv::getTickCount();
    std::cout << "Total cost Time: "
              << 1000 * (t1 - t0) / cv::getTickFrequency() << " ms"
              << std::endl;
    std::cout << "Per pixal cost Time: "
              << 1000000 * (t1 - t0) / cv::getTickFrequency() / (w * h) << " us"
              << std::endl;
    std::cout << "theta: " << theta_max * 57.3 << std::endl;

    ros::Duration delay(0.1);

    while(ros::ok())
    {
        pub_point.publish(imagePoint);
        delay.sleep();
    }

    return 0;
}
