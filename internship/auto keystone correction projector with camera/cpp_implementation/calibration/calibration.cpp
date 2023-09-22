#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/structured_light/graycodepattern.hpp>
#include <opencv2/core/utility.hpp>
#include <json/json.h>

namespace fs = std::filesystem;

void printNumpyWithIndent(const cv::Mat &mat, const std::string &indent) {
    std::cout << indent;
    std::cout << mat << std::endl;
}

void loadCameraParam(const std::string &json_file, cv::Mat &camP, cv::Mat &cam_dist) {
    std::ifstream file(json_file);
    Json::Value root;
    file >> root;
    cv::Mat P(3, 3, CV_64F);
    cv::Mat d(5, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            P.at<double>(i, j) = root["camera"]["P"][i * 3 + j].asDouble();
        }
    }
    for (int i = 0; i < 5; ++i) {
        d.at<double>(i) = root["camera"]["distortion"][i].asDouble();
    }
    camP = P.clone();
    cam_dist = d.clone();
}

void calibrate(const std::vector<std::string> &dirnames, const std::vector<std::vector<std::string>> &gc_fname_lists,
               cv::Size proj_shape, cv::Size chess_shape, double chess_block_size, int gc_step, int black_thr,
               int white_thr,
               cv::Mat &camP, cv::Mat &camD) {

    // Calculate the number of points and allocate memory for objps
    int num_points = chess_shape.width * chess_shape.height;
    std::vector<cv::Point3f> objps;

    // Fill objps with the grid of points
    for (int i = 0; i < chess_shape.height; ++i) {
        for (int j = 0; j < chess_shape.width; ++j) {
            objps.emplace_back(j * chess_block_size,i * chess_block_size,0);
        }
    }
    std::cout << "Calibrating ..." << std::endl;

    int gc_height = (proj_shape.height - 1) / gc_step + 1;
    int gc_width = (proj_shape.width - 1) / gc_step + 1;
    cv::Ptr<cv::structured_light::GrayCodePattern> graycode = cv::structured_light::GrayCodePattern ::create(gc_width,
                                                                                                            gc_height);
    graycode->setBlackThreshold(black_thr);
    graycode->setWhiteThreshold(white_thr);

    cv::Size cam_shape = cv::imread(gc_fname_lists[0][0], cv::IMREAD_GRAYSCALE).size();
    int patch_size_half = static_cast<int>(std::ceil(cam_shape.width / 180.0));

    std::cout << "  patch size: " << patch_size_half * 2 + 1 << std::endl;

    std::vector<std::vector<cv::Point2f>> cam_corners_list;
    std::vector<std::vector<cv::Point3f>> cam_objps_list;
    std::vector<std::vector<cv::Point2f>> cam_corners_list2;
    std::vector<std::vector<cv::Point3f>> proj_objps_list;
    std::vector<std::vector<cv::Point2f>> proj_corners_list;

    int dname_count = 0;
    for (const auto &dname: dirnames) {
        dname_count++;
        std::cout << "  checking '" << dname << "'" << std::endl;
        const auto &gc_filenames = gc_fname_lists[dname_count-1];
        if (gc_filenames.size() != graycode->getNumberOfPatternImages() + 2) {
            std::cout << "Error: Invalid number of images in '" << dname << "'" << std::endl;
            return;
        }

        std::vector<cv::Mat> imgs;
        for (const auto &fname: gc_filenames) {
            cv::Mat img = cv::imread(fname, cv::IMREAD_GRAYSCALE);
            if (cam_shape != img.size()) {
                std::cout << "Error: Image size of '" << fname << "' is mismatch" << std::endl;
                return;
            }
            imgs.push_back(img);
        }
        cv::Mat black_img = imgs.back();
        imgs.pop_back();
        cv::Mat white_img = imgs.back();
        imgs.pop_back();

        std::vector<cv::Point2f> cam_corners;
        bool found = cv::findChessboardCorners(white_img, chess_shape, cam_corners, cv::CALIB_CB_ADAPTIVE_THRESH);
        if (!found) {
            std::cout << "Error: Chessboard was not found in '" << gc_filenames[gc_filenames.size() - 2] << "'"
                      << std::endl;
            return;
        }
        cam_corners_list.push_back(cam_corners);
        cam_objps_list.push_back(objps);

        std::vector<cv::Point3f> proj_objps;
        std::vector<cv::Point2f> proj_corners;
        std::vector<cv::Point2f> cam_corners2;
        int count = 0;
        for (const auto &corner: cam_corners) {
            int c_x = static_cast<int>(std::round(corner.x));
            int c_y = static_cast<int>(std::round(corner.y));
            std::vector<cv::Point2f> src_points;
            std::vector<cv::Point2f> dst_points;
            for (int dx = -patch_size_half; dx <= patch_size_half; ++dx) {
                for (int dy = -patch_size_half; dy <= patch_size_half; ++dy) {
                    int x = c_x + dx;
                    int y = c_y + dy;
                    if (static_cast<int>(white_img.at<uchar>(y, x)) - static_cast<int>(black_img.at<uchar>(y, x)) <=
                        black_thr) {
                        continue;
                    }
                    bool err;
                    cv::Point proj_pix;
                    err = graycode->getProjPixel(imgs, x, y, proj_pix);
                    if (!err) {
                        src_points.emplace_back(x, y);
                        dst_points.push_back(proj_pix * gc_step);
                    }
                }
            }
            if (src_points.size() < patch_size_half * patch_size_half) {
                std::cout << "    Warning: Corner " << c_x << " " << c_y
                          << " was skipped because decoded pixels were too few (check your images and thresholds)"
                          << std::endl;
                continue;
            }
            cv::Mat H = cv::findHomography(src_points, dst_points);
            cv::Mat point = H * (cv::Mat_<double>(3, 1) << corner.x, corner.y, 1);
            cv::Point2f point_pix(point.at<double>(0) / point.at<double>(2), point.at<double>(1) / point.at<double>(2));
            proj_objps.push_back(objps[count]);
            proj_corners.push_back(point_pix);
            cam_corners2.push_back(corner);
            count++;
        }
        if (proj_corners.size() < 3) {
            std::cout << "Error: Too few corners were found in '" << dname << "' (less than 3)" << std::endl;
            return;
        }
        proj_objps_list.push_back(proj_objps);
        proj_corners_list.push_back(proj_corners);
        cam_corners_list2.push_back(cam_corners2);
    }

    std::cout << "Initial solution of camera's intrinsic parameters" << std::endl;
    std::vector<cv::Mat> cam_rvecs;
    std::vector<cv::Mat> cam_tvecs;
    cv::Mat cam_int;
    cv::Mat cam_dist;
    if (camP.empty()) {
        double rms = cv::calibrateCamera(cam_objps_list, cam_corners_list, cam_shape, cam_int, cam_dist, cam_rvecs, cam_tvecs);
        std::cout << "  RMS: " << rms << std::endl;
    } else {
        for (size_t i = 0; i < cam_objps_list.size(); ++i) {
            cv::Mat rvec, tvec;
            double rms = cv::solvePnP(cam_objps_list[i], cam_corners_list[i], camP, camD, rvec, tvec);
            cam_rvecs.push_back(rvec);
            cam_tvecs.push_back(tvec);
            std::cout << "  RMS: " << rms << std::endl;
            cam_int = camP.clone();
            cam_dist = camD.clone();
        }
    }
    std::cout << "  Intrinsic parameters:" << std::endl;
    printNumpyWithIndent(cam_int, "    ");
    std::cout << "  Distortion parameters:" << std::endl;
    printNumpyWithIndent(cam_dist, "    ");
    std::cout << std::endl;

    std::cout << "Initial solution of projector's parameters" << std::endl;
    cv::Mat proj_int, proj_dist;
    double rms = cv::calibrateCamera(proj_objps_list, proj_corners_list, proj_shape, proj_int, proj_dist, cam_rvecs, cam_tvecs);
    std::cout << "  RMS: " << rms << std::endl;
    std::cout << "  Intrinsic parameters:" << std::endl;
    printNumpyWithIndent(proj_int, "    ");
    std::cout << "  Distortion parameters:" << std::endl;
    printNumpyWithIndent(proj_dist, "    ");
    std::cout << std::endl;

    std::cout << "=== Result ===" << std::endl;
    cv::Mat cam_proj_rmat, cam_proj_tvec, E, F;
    rms = cv::stereoCalibrate(proj_objps_list, cam_corners_list2, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, cv::Size(), cam_proj_rmat, cam_proj_tvec, E, F);
    std::cout << "  RMS: " << rms << std::endl;
    std::cout << "  Camera intrinsic parameters:" << std::endl;
    printNumpyWithIndent(cam_int, "    ");
    std::cout << "  Camera distortion parameters:" << std::endl;
    printNumpyWithIndent(cam_dist, "    ");
    std::cout << "  Projector intrinsic parameters:" << std::endl;
    printNumpyWithIndent(proj_int, "    ");
    std::cout << "  Projector distortion parameters:" << std::endl;
    printNumpyWithIndent(proj_dist, "    ");
    std::cout << "  Rotation matrix / translation vector from camera to projector" << std::endl;
    std::cout << "  (They translate points from camera coord to projector coord):" << std::endl;
    printNumpyWithIndent(cam_proj_rmat, "    ");
    printNumpyWithIndent(cam_proj_tvec, "    ");

    cv::FileStorage fs("calibration_result.xml", cv::FileStorage::WRITE);
    fs << "img_shape" << cam_shape;
    fs << "rms" << rms;
    fs << "cam_int" << cam_int;
    fs << "cam_dist" << cam_dist;
    fs << "proj_int" << proj_int;
    fs << "proj_dist" << proj_dist;
    fs << "rotation" << cam_proj_rmat;
    fs << "translation" << cam_proj_tvec;
    fs.release();
}

int main() {
    cv::Size proj_shape(1080, 1920);
    cv::Size chess_shape(11, 8);
    double chess_block_size = 30.0;
    int gc_step = 16;
    int black_thr = 40;
    int white_thr = 5;

    std::string camera_param_file = "/Users/zhangzhouhao/Documents/Work/projector/python_implementation/calibration_tools/camera_config.json";

    std::vector<std::string> dirnames;
    for (const auto &entry: fs::directory_iterator("/Users/zhangzhouhao/Documents/Work/projector/python_implementation/calibration_tools/capture_old")) {
        if (entry.is_directory() && entry.path().filename().string().find("capture_") == 0) {
            dirnames.push_back(entry.path().string());
        }
    }
    if (dirnames.empty()) {
        std::cout << "Directories './capture_*' were not found" << std::endl;
        return 1;
    }
    // Sort dirnames
    std::sort(dirnames.begin(), dirnames.end());
    std::cout << "Searching input files ..." << std::endl;
    std::vector<std::string> used_dirnames;
    std::vector<std::vector<std::string>> gc_fname_lists;
    for (const std::string &dname: dirnames) {
        std::vector<std::string> gc_fnames;
        for (const auto &entry: fs::directory_iterator(dname)) {
            if (entry.is_regular_file() && entry.path().filename().string().find("graycode_") == 0) {
                gc_fnames.push_back(entry.path().string());
            }
        }
        if (!gc_fnames.empty()) {
            used_dirnames.push_back(dname);
            // Sort gc_fnames
            std::sort(gc_fnames.begin(), gc_fnames.end());
            gc_fname_lists.push_back(gc_fnames);
            std::cout << " '" << dname << "' was found" << std::endl;
        }
    }

    cv::Mat camP, cam_dist;
    loadCameraParam(camera_param_file, camP, cam_dist);
    std::cout << "Load camera parameters" << std::endl;
    std::cout << "camP: " << camP << std::endl;
    std::cout << "cam_dist: " << cam_dist << std::endl;

    calibrate(dirnames, gc_fname_lists, proj_shape, chess_shape, chess_block_size, gc_step, black_thr, white_thr, camP,
              cam_dist);

    return 0;
}
