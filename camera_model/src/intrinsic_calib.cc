#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <camera_model/calib/CameraCalibration.h>
#include <camera_model/chessboard/Chessboard.h>
#include <camera_model/gpl/gpl.h>

int
main(int argc, char** argv)
{
    cv::Size    boardSize;
    float       squareSize;
    std::string inputDir;
    std::string cameraModel;
    std::string cameraName;
    std::string prefix;
    std::string fileExtension;
    bool        useOpenCV;
    bool        viewResults;
    bool        verbose;
    int         diff_num;

    //========= Handling Program options =========

    /* clang-format off */
    using namespace boost::program_options;
    boost::program_options::options_description desc("Allowed options.\n Ask GAO Wenliang if there is any possible questions.\n");
    desc.add_options()
        ("help", "produce help message")
        ("width,w", value<int>(&boardSize.width)->default_value(8),"Number of inner corners on the chessboard pattern in x direction")
        ("height,h", value<int>(&boardSize.height)->default_value(12),"Number of inner corners on the chessboard pattern in y direction")
        ("size,s", value<float>(&squareSize)->default_value(7.f),"Size of one square in mm")
        ("input,i",value<std::string>(&inputDir)->default_value("calibrationdata"),"Input directory containing chessboard images")
        ("prefix,p", value<std::string>(&prefix)->default_value("left-"),"Prefix of images")
        ("file-extension,e",value<std::string>(&fileExtension)->default_value(".png"),"File extension of images")
        ("camera-model", value<std::string>(&cameraModel)->default_value("mei"),"Camera model: kannala-brandt | fov | scaramuzza | mei | pinhole | myfisheye")
        ("camera-name", value<std::string>(&cameraName)->default_value("camera"),"Name of camera")
        ("opencv", value<bool>(&useOpenCV)->default_value(true),"Use OpenCV to detect corners")
        ("view-results", value<bool>(&viewResults)->default_value(true),"View results")
        ("verbose,v", value<bool>(&verbose)->default_value(true),    "Verbose output")
        ("num,n", value<int>(&diff_num)->default_value(1000),"polynomial fast computer diff number");

    boost::program_options::positional_options_description pdesc;
    pdesc.add("input", 1);

    boost::program_options::variables_map vm;
    boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
        .options(desc)
        .positional(pdesc)
        .run(),
      vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 1;
    }

    if (!boost::filesystem::exists(inputDir) &&
        !boost::filesystem::is_directory(inputDir))
    {
        std::cerr << "# ERROR: Cannot find input directory " << inputDir << "."
                  << std::endl;
        return 1;
    }

    camera_model::Camera::ModelType modelType;
    if (boost::iequals(cameraModel, "kannala-brandt"))
    {
        modelType = camera_model::Camera::KANNALA_BRANDT;
    }
    else if (boost::iequals(cameraModel, "mei"))
    {
        modelType = camera_model::Camera::MEI;
    }
    else if (boost::iequals(cameraModel, "pinhole"))
    {
        modelType = camera_model::Camera::PINHOLE;
    }
    else if (boost::iequals(cameraModel, "scaramuzza"))
    {
        modelType = camera_model::Camera::SCARAMUZZA;
    }
    else if (boost::iequals(cameraModel, "myfisheye"))
    {
        modelType = camera_model::Camera::POLYFISHEYE;
    }
    else if (boost::iequals(cameraModel, "fov"))
    {
        modelType = camera_model::Camera::FOV;
    }
    else
    {
        std::cerr << "# ERROR: Unknown camera model: " << cameraModel
                  << std::endl;
        return 1;
    }

    switch (modelType)
    {
        case camera_model::Camera::KANNALA_BRANDT:
            std::cout << "# INFO: Camera model: Kannala-Brandt" << std::endl;
            break;
        case camera_model::Camera::MEI:
            std::cout << "# INFO: Camera model: Mei" << std::endl;
            break;
        case camera_model::Camera::PINHOLE:
            std::cout << "# INFO: Camera model: Pinhole" << std::endl;
            break;
        case camera_model::Camera::SCARAMUZZA:
            std::cout << "# INFO: Camera model: Scaramuzza-Omnidirect"
                      << std::endl;
            break;
        case camera_model::Camera::FOV:
            std::cout << "# INFO: Camera model: FOV camera model"
                      << std::endl;
            break;
        case camera_model::Camera::POLYFISHEYE:
            std::cout
              << "# INFO: Camera model: GaoWenliang's polynomial fisheye model"
              << std::endl;
            break;
    }

    // look for images in input directory
    std::vector<std::string>              imageFilenames;
    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(inputDir);
         itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (!boost::filesystem::is_regular_file(itr->status()))
        {
            continue;
        }

        std::string filename = itr->path().filename().string();

        // check if prefix matches
        if (!prefix.empty())
        {
            if (filename.compare(0, prefix.length(), prefix) != 0)
            {
                continue;
            }
        }

        // check if file extension matches
        if (filename.compare(filename.length() - fileExtension.length(),
                             fileExtension.length(), fileExtension) != 0)
        {
            continue;
        }

        imageFilenames.push_back(itr->path().string());

        if (verbose)
        {
            std::cerr << "# INFO: Adding " << imageFilenames.back()
                      << std::endl;
        }
    }

    if (imageFilenames.empty())
    {
        std::cerr << "# ERROR: No chessboard images found." << std::endl;
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: # images: " << imageFilenames.size() << std::endl;
    }

    cv::Mat image = cv::imread(imageFilenames.front(), -1);
    const cv::Size frameSize = image.size();

    // TODO need to change mode type
    camera_model::CameraCalibration calibration(
      modelType, cameraName, frameSize, boardSize, squareSize);
    calibration.setVerbose(verbose);

    std::vector<bool> chessboardFound(imageFilenames.size(), false);
    for (size_t i = 0; i < imageFilenames.size(); ++i)
    {
        image = cv::imread(imageFilenames.at(i), -1);

        camera_model::Chessboard chessboard(boardSize, image);

        chessboard.findCorners(useOpenCV);
        if (chessboard.cornersFound())
        {
            if (verbose)
            {
                std::cerr << "# INFO: Detected chessboard in image " << i + 1
                          << ", " << imageFilenames.at(i) << std::endl;
            }

            calibration.addChessboardData(chessboard.getCorners());

            cv::Mat sketch;
            chessboard.getSketch().copyTo(sketch);

            cv::namedWindow("Image", cv::WINDOW_NORMAL);
            cv::imshow("Image", sketch);
            cv::waitKey(10);
        }
        else if (verbose)
        {
            std::cout << "\033[31;47;1m" << "# INFO: Did not detect chessboard in image: "
                      << imageFilenames.at(i) << "\033[0m" << std::endl;
        }
        chessboardFound.at(i) = chessboard.cornersFound();
    }
    cv::destroyWindow("Image");

    if (calibration.sampleCount() < 10)
    {
        std::cerr << "# ERROR: Insufficient number of detected chessboards."
                  << std::endl;
        return 1;
    }

    if (verbose)
    {
        std::cerr << "# INFO: Calibrating..." << std::endl;
    }

    double startTime = camera_model::timeInSeconds();

    std::cout << " calibrate start." << std::endl;
    calibration.calibrate();

    std::cout << " calibrate done." << std::endl;

    calibration.writeParams(cameraName + "_camera_calib.yaml");
    calibration.writeChessboardData(cameraName + "_chessboard_data.dat");

    if (verbose)
    {
        std::cout << "# INFO: Calibration took a total time of " << std::fixed
                  << std::setprecision(3)
                  << camera_model::timeInSeconds() - startTime << " sec.\n";
    }

    if (verbose)
    {
        std::cerr << "# INFO: Wrote calibration file to "
                  << cameraName + "_camera_calib.yaml" << std::endl;
    }

    if (viewResults)
    {
        std::vector<cv::Mat>     cbImages;
        std::vector<std::string> cbImageFilenames;

        for (size_t i = 0; i < imageFilenames.size(); ++i)
        {
            if (!chessboardFound.at(i))
                continue;

            cbImages.push_back(cv::imread(imageFilenames.at(i), -1));
            cbImageFilenames.push_back(imageFilenames.at(i));
        }
        std::cout << "\033[32;40;1m" << "# INFO: Used image num: "
                  << cbImages.size() << "\033[0m" << std::endl;

        // visualize observed and reprojected points
        calibration.drawResults(cbImages);

        for (size_t i = 0; i < cbImages.size(); ++i)
        {
            cv::putText(cbImages.at(i), cbImageFilenames.at(i),
                        cv::Point(10, 20), cv::FONT_HERSHEY_COMPLEX, 0.5,
                        cv::Scalar(0, 0, 255), 1, CV_AA);

            cv::namedWindow("Image", cv::WINDOW_NORMAL);
            cv::imshow("Image", cbImages.at(i));
            cv::waitKey(0);
        }

    }

    return 0;
}
