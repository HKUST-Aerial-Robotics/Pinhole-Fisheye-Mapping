#include <camera_model/camera_models/PolyFisheyeCamera.h>

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <camera_model/gpl/gpl.h>
#include <camera_model/code_utils/math_utils/math_utils.h>

namespace camera_model
{

PolyFisheyeCamera::PolyFisheyeCamera()
  : m_inv_K11(1.0)
  , m_inv_K12(0.0)
  , m_inv_K13(0.0)
  , m_inv_K22(1.0)
  , m_inv_K23(0.0)
{
    poly = new math_utils::Polynomial(FISHEYE_POLY_ORDER);

    poly->setPolyCoeff(0, 0.0);
    poly->setPolyCoeff(1, 1.0);
}

PolyFisheyeCamera::PolyFisheyeCamera(const std::string& cameraName,
                                     int imageWidth, int imageHeight, double k2,
                                     double k3, double k4, double k5, double k6,
                                     double k7, double A11, double A12,
                                     double A22, double u0, double v0,
                                     int isFast)
  : mParameters(cameraName, imageWidth, imageHeight, k2, k3, k4, k5, k6, k7,
                A11, A12, A22, u0, v0, isFast)
{
    eigen_utils::Vector coeff(FISHEYE_POLY_ORDER + 1);
    coeff << 0.0, 1.0, k2, k3, k4, k5, k6, k7;

    poly = new math_utils::Polynomial(FISHEYE_POLY_ORDER);
    poly->setPolyCoeff(coeff);

    if (mParameters.isFast() == 1)
    {
        eigen_utils::Vector coeff(FISHEYE_POLY_ORDER + 1);
        coeff << 0.0, 1.0, mParameters.k2(), mParameters.k3(), mParameters.k4(),
          mParameters.k5(), mParameters.k6(), mParameters.k7();

//        fastCalc = new FastCalcPOLY(coeff,
//                                    (double) mParameters.maxIncidentAngle());
          fastCalc = new FastCalcTABLE(coeff, mParameters.numDiff(), (double)
                                       mParameters.maxIncidentAngle());
    }
}

PolyFisheyeCamera::PolyFisheyeCamera(
  const PolyFisheyeCamera::Parameters& params)
  : mParameters(params)
{
    eigen_utils::Vector coeff(FISHEYE_POLY_ORDER + 1);
    coeff << 0.0, 1.0, params.k2(), params.k3(), params.k4(), params.k5(),
      params.k6(), params.k7();

    poly = new math_utils::Polynomial(FISHEYE_POLY_ORDER);
    poly->setPolyCoeff(coeff);

    calcKinvese(params.A11(), params.A12(), params.A22(), params.u0(),
                params.v0());

    if (mParameters.isFast() == 1)
    {
        eigen_utils::Vector coeff(FISHEYE_POLY_ORDER + 1);
        coeff << 0.0, 1.0, mParameters.k2(), mParameters.k3(), mParameters.k4(),
          mParameters.k5(), mParameters.k6(), mParameters.k7();

//        fastCalc = new FastCalcPOLY(coeff,
//                                    (double) mParameters.maxIncidentAngle());
          fastCalc = new FastCalcTABLE(coeff, mParameters.numDiff(), (double)
                                       mParameters.maxIncidentAngle());
    }
}

Camera::ModelType
PolyFisheyeCamera::modelType() const
{
    return mParameters.modelType();
}

const std::string&
PolyFisheyeCamera::cameraName() const
{
    return mParameters.cameraName();
}

int
PolyFisheyeCamera::imageWidth() const
{
    return mParameters.imageWidth();
}

int
PolyFisheyeCamera::imageHeight() const
{
    return mParameters.imageHeight();
}

void
PolyFisheyeCamera::spaceToPlane(const Eigen::Vector3d& P,
                                Eigen::Vector2d&       p) const
{
    double theta = acos(P(2) / P.norm());
    double phi   = atan2(P(1), P(0));

    // TODO
    Eigen::Vector2d p_u;
    if (mParameters.isFast() == 1)
    {
        p_u = fastCalc->r(theta) * Eigen::Vector2d(cos(phi), sin(phi));
        //    std::cout<< " p_u " << p_u.transpose()<<std::endl;
    }
    else
    {
        p_u = r(mParameters.k2(), mParameters.k3(), mParameters.k4(),
                mParameters.k5(), mParameters.k6(), mParameters.k7(), theta) *
              Eigen::Vector2d(cos(phi), sin(phi));
        //  std::cout<< "theta "<<theta<< " r " << fastCalc->r( theta)
        //  <<std::endl;
    }
    // Apply generalised projection matrix
    p(0) = mParameters.A11() * p_u(0) + mParameters.A12() * p_u(1) +
           mParameters.u0();
    p(1) = mParameters.A22() * p_u(1) + mParameters.v0();
}

/**
 * \brief Project a 3D point to the image plane and calculate Jacobian
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void
PolyFisheyeCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                                Eigen::Matrix<double, 2, 3>& J) const
{
    double theta = acos(P(2) / P.norm());
    double phi   = atan2(P(1), P(0));

    // TODO
    Eigen::Vector2d p_u;
    if (mParameters.isFast() == 1)
        p_u = fastCalc->r(theta) * Eigen::Vector2d(cos(phi), sin(phi));
    else
        p_u = r(mParameters.k2(), mParameters.k3(), mParameters.k4(),
                mParameters.k5(), mParameters.k6(), mParameters.k7(), theta) *
              Eigen::Vector2d(cos(phi), sin(phi));

    // Apply generalised projection matrix
    p << mParameters.A11() * p_u(0) + mParameters.A12() * p_u(1) +
           mParameters.u0(),
      mParameters.A22() * p_u(1) + mParameters.v0();
}

double PolyFisheyeCamera::getMaxIncidentAngle()
{
    return fastCalc->getMaxIncidentAngle();
}
double PolyFisheyeCamera::get_diffangle() const
{
    return fastCalc->getDiffAngle();
}
double PolyFisheyeCamera::get_diffr() const
{
    return fastCalc->getDiffR();
}
eigen_utils::Matrix PolyFisheyeCamera::getMatAngleToR()
{
    return fastCalc->getMatAngleToR();
}
eigen_utils::Matrix PolyFisheyeCamera::getMatRToAngle()
{
    return fastCalc->getMatRToAngle();
}

void
PolyFisheyeCamera::estimateIntrinsics(
  const cv::Size&                              boardSize,
  const std::vector<std::vector<cv::Point3f>>& objectPoints,
  const std::vector<std::vector<cv::Point2f>>& imagePoints)
{
    Parameters params = getParameters();

    double u0 = params.imageWidth() / 2.0;
    double v0 = params.imageHeight() / 2.0;

    double minReprojErr = std::numeric_limits<double>::max();

    std::vector<cv::Mat> rvecs, tvecs;
    rvecs.assign(objectPoints.size(), cv::Mat());
    tvecs.assign(objectPoints.size(), cv::Mat());

    params.k2() = 0.0;
    params.k3() = 0.0;
    params.k4() = 0.0;
    params.k5() = 0.0;
    params.k6() = 0.0;
    params.k7() = 0.0;
    params.u0() = u0;
    params.v0() = v0;

    // Initialize focal length
    // C. Hughes, P. Denny, M. Glavin, and E. Jones,
    // Equidistant Fish-Eye Calibration and Rectification by Vanishing Point
    // Extraction, PAMI 2010
    // Find circles from rows of chessboard corners, and for each pair
    // of circles, find vanishing points: v1 and v2.
    // f = ||v1 - v2|| / PI;
    double f0 = 0.0;
    for (size_t i = 0; i < imagePoints.size(); ++i)
    {
        std::vector<Eigen::Vector2d> center(boardSize.height);
        double                       radius[boardSize.height];
        for (int r = 0; r < boardSize.height; ++r)
        {
            std::vector<cv::Point2d> circle;
            for (int c = 0; c < boardSize.width; ++c)
            {
                circle.push_back(imagePoints.at(i).at(r * boardSize.width + c));
            }

            fitCircle(circle, center[r](0), center[r](1), radius[r]);
        }

        for (int j = 0; j < boardSize.height; ++j)
        {
            for (int k = j + 1; k < boardSize.height; ++k)
            {
                // find distance between pair of vanishing points which
                // correspond to intersection points of 2 circles
                std::vector<cv::Point2d> ipts;
                ipts = intersectCircles(center[j](0), center[j](1), radius[j],
                                        center[k](0), center[k](1), radius[k]);

                if (ipts.size() < 2)
                {
                    continue;
                }

                double f = cv::norm(ipts.at(0) - ipts.at(1)) / M_PI;

                params.A11() = f;
                params.A22() = f;

                setParameters(params);

                for (size_t l = 0; l < objectPoints.size(); ++l)
                {
                    estimateExtrinsics(objectPoints.at(l), imagePoints.at(l),
                                       rvecs.at(l), tvecs.at(l));
                }

                double reprojErr = reprojectionError(
                  objectPoints, imagePoints, rvecs, tvecs, cv::noArray());

                if (reprojErr < minReprojErr)
                {
                    minReprojErr = reprojErr;
                    f0           = f;
                }
            }
        }
    }

    if (f0 <= 0.0 && minReprojErr >= std::numeric_limits<double>::max())
    {
        std::cout << "[" << params.cameraName() << "] "
                  << "# INFO: kannala-Brandt model fails with given data. "
                  << std::endl;

        return;
    }

    params.A11() = f0;
    params.A22() = f0;

    setParameters(params);
}

void
PolyFisheyeCamera::liftSphere(const Eigen::Vector2d& p,
                              Eigen::Vector3d&       P) const
{
    liftProjective(p, P);
}

void
PolyFisheyeCamera::rayToPlane(const Ray& ray, Eigen::Vector2d& p) const
{
    Eigen::Vector2d p_u =
      r(mParameters.k2(), mParameters.k3(), mParameters.k4(), mParameters.k5(),
        mParameters.k6(), mParameters.k7(), ray.theta()) *
      Eigen::Vector2d(cos(ray.phi()), sin(ray.phi()));

    p(0) = mParameters.A11() * p_u(0) + mParameters.A12() * p_u(1) +
           mParameters.u0();
    p(1) = mParameters.A22() * p_u(1) + mParameters.v0();
}

void
PolyFisheyeCamera::liftProjectiveToRay(const Eigen::Vector2d& p, Ray& ray) const
{
    // TODO
    if (mParameters.isFast() == 1)
    {
        fastCalc->backprojectSymmetric(
          Eigen::Vector2d(m_inv_K11 * p(0) + m_inv_K12 * p(1) + m_inv_K13,
                          m_inv_K22 * p(1) + m_inv_K23),
          ray.theta(), ray.phi());
    }
    else
    {
        backprojectSymmetric(
          Eigen::Vector2d(m_inv_K11 * p(0) + m_inv_K12 * p(1) + m_inv_K13,
                          m_inv_K22 * p(1) + m_inv_K23),
          ray.theta(), ray.phi());
    }
}

double PolyFisheyeCamera::get_theta(int x, int y)
{
    Eigen::Vector2d p;
    p(0) = x; p(1) = y;

    // Lift points to normalised plane
    double theta, phi;
    // Obtain a projective ray

    // TODO
    if (mParameters.isFast() == 1)
    {
        fastCalc->backprojectSymmetric(
          Eigen::Vector2d(m_inv_K11 * p(0) + m_inv_K12 * p(1) + m_inv_K13,
                          m_inv_K22 * p(1) + m_inv_K23),
          theta, phi);
    }
    else
    {
        backprojectSymmetric(
          Eigen::Vector2d(m_inv_K11 * p(0) + m_inv_K12 * p(1) + m_inv_K13,
                          m_inv_K22 * p(1) + m_inv_K23),
          theta, phi);
    }

    return theta;
}

void
PolyFisheyeCamera::liftProjective(const Eigen::Vector2d& p,
                                  Eigen::Vector3d&       P) const
{
    // Lift points to normalised plane
    double theta, phi;
    // Obtain a projective ray

    // TODO
    if (mParameters.isFast() == 1)
    {
        fastCalc->backprojectSymmetric(
          Eigen::Vector2d(m_inv_K11 * p(0) + m_inv_K12 * p(1) + m_inv_K13,
                          m_inv_K22 * p(1) + m_inv_K23),
          theta, phi);
    }
    else
    {
        backprojectSymmetric(
          Eigen::Vector2d(m_inv_K11 * p(0) + m_inv_K12 * p(1) + m_inv_K13,
                          m_inv_K22 * p(1) + m_inv_K23),
          theta, phi);
    }
    P =
      Eigen::Vector3d(cos(phi) * sin(theta),
                      sin(phi) * sin(theta),
                      cos(theta));
}

void
PolyFisheyeCamera::undistToPlane(const Eigen::Vector2d& p_u,
                                 Eigen::Vector2d&       p) const
{
}

cv::Mat
PolyFisheyeCamera::initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                           float fx, float fy,
                                           cv::Size imageSize, float cx,
                                           float cy, cv::Mat rmat) const
{
    if (imageSize == cv::Size(0, 0))
    {
        imageSize =
          cv::Size(mParameters.imageWidth(), mParameters.imageHeight());
    }

    cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

    Eigen::Matrix3f K_rect;

    if (cx == -1.0f && cy == -1.0f)
    {
        K_rect << fx, 0, imageSize.width / 2,
            0, fy, imageSize.height / 2,
            0, 0, 1;
    }
    else
    {
        K_rect << fx, 0, cx,
            0, fy, cy,
            0, 0, 1;
    }

    if (fx == -1.0f || fy == -1.0f)
    {
        K_rect(0, 0) = mParameters.A11();
        K_rect(0, 1) = mParameters.A12();
        K_rect(1, 1) = mParameters.A22();
    }

    Eigen::Matrix3f K_rect_inv = K_rect.inverse();

    Eigen::Matrix3f R, R_inv;
    cv::cv2eigen(rmat, R);
    R_inv = R.inverse();

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            Eigen::Vector3f xo;
            xo << u, v, 1;

            // TODO FIXME
            Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

            Eigen::Vector2d p;
            spaceToPlane(uo.cast<double>(), p);

            mapX.at<float>(v, u) = p(0);
            mapY.at<float>(v, u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);

    cv::Mat K_rect_cv;
    cv::eigen2cv(K_rect, K_rect_cv);
    return K_rect_cv;
}

int
PolyFisheyeCamera::parameterCount() const
{
    return FISHEYE_PARAMS_NUM;
}

const PolyFisheyeCamera::Parameters&
PolyFisheyeCamera::getParameters() const
{
    return mParameters;
}

void
PolyFisheyeCamera::setParameters(
  const PolyFisheyeCamera::Parameters& parameters)
{
    mParameters = parameters;

    eigen_utils::Vector coeff(FISHEYE_POLY_ORDER + 1);
    coeff << 0.0, 1.0, mParameters.k2(), mParameters.k3(), mParameters.k4(), mParameters.k5(),
      mParameters.k6(), mParameters.k7();

    delete poly;
    poly = new math_utils::Polynomial(FISHEYE_POLY_ORDER);
    poly->setPolyCoeff(coeff);

    calcKinvese(parameters.A11(), parameters.A12(), parameters.A22(),
                parameters.u0(), parameters.v0());

    if (mParameters.isFast() == 1)
    {
        eigen_utils::Vector coeff_fast(FISHEYE_POLY_ORDER + 1);
        coeff_fast << 0.0, 1.0, mParameters.k2(), mParameters.k3(), mParameters.k4(),
          mParameters.k5(), mParameters.k6(), mParameters.k7();

//        if(fastCalc != NULL)
//          delete fastCalc;
//        fastCalc = new FastCalcPOLY(coeff_fast,
//                                    (double) mParameters.maxIncidentAngle());
          fastCalc = new FastCalcTABLE(coeff_fast, mParameters.numDiff(),
                                       (double)mParameters.maxIncidentAngle());
    }
}

void
PolyFisheyeCamera::readParameters(const std::vector<double>& parameterVec)
{
    if (int(parameterVec.size()) != parameterCount())
    {
        return;
    }

    Parameters params = getParameters();

    params.k2()  = parameterVec.at(0);
    params.k3()  = parameterVec.at(1);
    params.k4()  = parameterVec.at(2);
    params.k5()  = parameterVec.at(3);
    params.k6()  = parameterVec.at(4);
    params.k7()  = parameterVec.at(5);
    params.A11() = parameterVec.at(6);
    params.A12() = parameterVec.at(7);
    params.A22() = parameterVec.at(8);
    params.u0()  = parameterVec.at(9);
    params.v0()  = parameterVec.at(10);

    setParameters(params);
}

void
PolyFisheyeCamera::writeParameters(std::vector<double>& parameterVec) const
{
    parameterVec.resize(parameterCount());

    parameterVec.at(0)  = mParameters.k2();
    parameterVec.at(1)  = mParameters.k3();
    parameterVec.at(2)  = mParameters.k4();
    parameterVec.at(3)  = mParameters.k5();
    parameterVec.at(4)  = mParameters.k6();
    parameterVec.at(5)  = mParameters.k7();
    parameterVec.at(6)  = mParameters.A11();
    parameterVec.at(7)  = mParameters.A12();
    parameterVec.at(8)  = mParameters.A22();
    parameterVec.at(9)  = mParameters.u0();
    parameterVec.at(10) = mParameters.v0();
}

void
PolyFisheyeCamera::writeParametersToYamlFile(const std::string& filename) const
{
    mParameters.writeToYamlFile(filename);
}

std::string
PolyFisheyeCamera::parametersToString() const
{
    std::ostringstream oss;
    oss << mParameters;

    return oss.str();
}

void
PolyFisheyeCamera::backprojectSymmetric(const Eigen::Vector2d& p_u,
                                        double& theta, double& phi) const
{
    double r = p_u.norm();

    if (r < 1e-10)
        phi = 0.0;
    else
        phi = atan2(p_u(1), p_u(0));

    eigen_utils::Vector vec_root;
    theta = poly->getOneRealRoot(r, 0.0, MAX_INCIDENT_ANGLE_DEGREE / RAD2DEG);
    //  vec_root = poly->getRealRoot(r, 0.0, MAX_INCIDENT_ANGLE_DEGREE/RAD2DEG);

    //  if(theta == 0)
    //    std::cout << " r " << r << " theta " << theta  << std::endl;
    //  if(vec_root.size() == 1)
    //    theta = vec_root(0);
    //  else
    //    theta = 0;
}

bool
PolyFisheyeCamera::calcKinvese(double a11, double a12, double a22, double u0,
                               double v0)
{
    eigen_utils::Matrix K(3, 3);
    K << a11, a12, u0, 0, a22, v0, 0, 0, 1;

    eigen_utils::Matrix K_inv = K.inverse();
    //  std::cout << " K file: "<< K << std::endl;
    //  std::cout << " K_inv file: "<< K_inv << std::endl;

    // Inverse camera projection matrix parameters
    m_inv_K11 = K_inv(0, 0);
    m_inv_K12 = K_inv(0, 1);
    m_inv_K13 = K_inv(0, 2);
    m_inv_K22 = K_inv(1, 1);
    m_inv_K23 = K_inv(1, 2);
}

void
PolyFisheyeCamera::setPoly(math_utils::Polynomial* value)
{
    poly = value;
}

PolyFisheyeCamera::FastCalcTABLE*
PolyFisheyeCamera::getFastCalc()
{
    return fastCalc;
}

double
PolyFisheyeCamera::getInv_K23() const
{
    return m_inv_K23;
}

double
PolyFisheyeCamera::getInv_K22() const
{
    return m_inv_K22;
}

double
PolyFisheyeCamera::getInv_K13() const
{
    return m_inv_K13;
}

double
PolyFisheyeCamera::getInv_K12() const
{
    return m_inv_K12;
}

double
PolyFisheyeCamera::getInv_K11() const
{
    return m_inv_K11;
}

math_utils::Polynomial*
PolyFisheyeCamera::getPoly() const
{
    return poly;
}

PolyFisheyeCamera::Parameters::Parameters()
  : Camera::Parameters(POLYFISHEYE)
  , m_k2(0.0)
  , m_k3(0.0)
  , m_k4(0.0)
  , m_k5(0.0)
  , m_k6(0.0)
  , m_k7(0.0)
  , m_A11(1.0)
  , m_A12(0.0)
  , m_A22(1.0)
  , m_u0(0.0)
  , m_v0(0.0)
  , m_isFast(0)
  , m_numDiff(FAST_NUM_DEFAULT)
  , m_maxIncidentAngle(FAST_MAX_INCIDENT_ANGLE)
{
}

PolyFisheyeCamera::Parameters::Parameters(const std::string& cameraName, int w,
                                          int h, double k2, double k3,
                                          double k4, double k5, double k6,
                                          double k7, double A11, double A12,
                                          double A22, double u0, double v0,
                                          int isFast)
  : Camera::Parameters(POLYFISHEYE, cameraName, w, h)
  , m_k2(k2)
  , m_k3(k3)
  , m_k4(k4)
  , m_k5(k5)
  , m_k6(k6)
  , m_k7(k7)
  , m_A11(A11)
  , m_A12(A12)
  , m_A22(A22)
  , m_u0(u0)
  , m_v0(v0)
  , m_isFast(isFast)
  , m_numDiff(FAST_NUM_DEFAULT)
  , m_maxIncidentAngle(FAST_MAX_INCIDENT_ANGLE)
{
}

double&
PolyFisheyeCamera::Parameters::k2()
{
    return m_k2;
}

double&
PolyFisheyeCamera::Parameters::k3()
{
    return m_k3;
}

double&
PolyFisheyeCamera::Parameters::k4()
{
    return m_k4;
}

double&
PolyFisheyeCamera::Parameters::k5()
{
    return m_k5;
}

double&
PolyFisheyeCamera::Parameters::k6()
{
    return m_k6;
}

double&
PolyFisheyeCamera::Parameters::k7()
{
    return m_k7;
}

double&
PolyFisheyeCamera::Parameters::A11()
{
    return m_A11;
}

double&
PolyFisheyeCamera::Parameters::A12()
{
    return m_A12;
}

double&
PolyFisheyeCamera::Parameters::A22()
{
    return m_A22;
}

double&
PolyFisheyeCamera::Parameters::u0()
{
    return m_u0;
}

double&
PolyFisheyeCamera::Parameters::v0()
{
    return m_v0;
}

int&
PolyFisheyeCamera::Parameters::numDiff()
{
    return m_numDiff;
}

int&
PolyFisheyeCamera::Parameters::maxIncidentAngle()
{
    return m_maxIncidentAngle;
}

int&
PolyFisheyeCamera::Parameters::isFast()
{
    return m_isFast;
}

double
PolyFisheyeCamera::Parameters::k2() const
{
    return m_k2;
}

double
PolyFisheyeCamera::Parameters::k3() const
{
    return m_k3;
}

double
PolyFisheyeCamera::Parameters::k4() const
{
    return m_k4;
}

double
PolyFisheyeCamera::Parameters::k5() const
{
    return m_k5;
}

double
PolyFisheyeCamera::Parameters::k6() const
{
    return m_k6;
}

double
PolyFisheyeCamera::Parameters::k7() const
{
    return m_k7;
}

double
PolyFisheyeCamera::Parameters::A11() const
{
    return m_A11;
}

double
PolyFisheyeCamera::Parameters::A12() const
{
    return m_A12;
}

double
PolyFisheyeCamera::Parameters::A22() const
{
    return m_A22;
}

double
PolyFisheyeCamera::Parameters::u0() const
{
    return m_u0;
}

double
PolyFisheyeCamera::Parameters::v0() const
{
    return m_v0;
}

int
PolyFisheyeCamera::Parameters::isFast() const
{
    return m_isFast;
}

int
PolyFisheyeCamera::Parameters::numDiff() const
{
    return m_numDiff;
}

int
PolyFisheyeCamera::Parameters::maxIncidentAngle() const
{
    return m_maxIncidentAngle;
}

bool
PolyFisheyeCamera::Parameters::readFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (sModelType.compare("POLYFISHEYE") != 0)
        {
            return false;
        }
    }

    m_modelType = POLYFISHEYE;
    fs["camera_name"] >> m_cameraName;
    m_imageWidth  = static_cast<int>(fs["image_width"]);
    m_imageHeight = static_cast<int>(fs["image_height"]);

    cv::FileNode n = fs["projection_parameters"];

    m_k2  = static_cast<double>(n["k2"]);
    m_k3  = static_cast<double>(n["k3"]);
    m_k4  = static_cast<double>(n["k4"]);
    m_k5  = static_cast<double>(n["k5"]);
    m_k6  = static_cast<double>(n["k6"]);
    m_k7  = static_cast<double>(n["k7"]);
    m_A11 = static_cast<double>(n["A11"]);
    m_A12 = static_cast<double>(n["A12"]);
    m_A22 = static_cast<double>(n["A22"]);
    m_u0  = static_cast<double>(n["u0"]);
    m_v0  = static_cast<double>(n["v0"]);

    m_isFast           = static_cast<int>(n["isFast"]);
    m_numDiff          = static_cast<int>(n["numDiff"]);
    m_maxIncidentAngle = static_cast<int>(n["maxIncidentAngle"]);

    return true;
}

void
PolyFisheyeCamera::Parameters::writeToYamlFile(
  const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "model_type"
       << "POLYFISHEYE";
    fs << "camera_name" << m_cameraName;
    fs << "image_width" << m_imageWidth;
    fs << "image_height" << m_imageHeight;

    // projection:  k2, k3, k4, k5, k6, k7, A11, A22, u0, v0
    fs << "projection_parameters";
    fs << "{";

    fs << "k2" << m_k2;
    fs << "k3" << m_k3;
    fs << "k4" << m_k4;
    fs << "k5" << m_k5;
    fs << "k6" << m_k6;
    fs << "k7" << m_k7;
    fs << "A11" << m_A11;
    fs << "A12" << m_A12;
    fs << "A22" << m_A22;
    fs << "u0" << m_u0;
    fs << "v0" << m_v0;
    fs << "isFast" << m_isFast;
    fs << "numDiff" << m_numDiff;
    fs << "maxIncidentAngle" << m_maxIncidentAngle;

    fs << "}";

    fs.release();
}

PolyFisheyeCamera::Parameters&
PolyFisheyeCamera::Parameters::operator=(
  const PolyFisheyeCamera::Parameters& other)
{
    if (this != &other)
    {
        m_modelType   = other.m_modelType;
        m_cameraName  = other.m_cameraName;
        m_imageWidth  = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;

        m_k2 = other.m_k2;
        m_k3 = other.m_k3;
        m_k4 = other.m_k4;
        m_k5 = other.m_k5;
        m_k6 = other.m_k6;
        m_k7 = other.m_k7;

        m_A11 = other.m_A11;
        m_A12 = other.m_A12;
        m_A22 = other.m_A22;

        m_u0 = other.m_u0;
        m_v0 = other.m_v0;

        m_isFast           = other.m_isFast;
        m_numDiff          = other.m_numDiff;
        m_maxIncidentAngle = other.m_maxIncidentAngle;
    }

    return *this;
}

std::ostream&
operator<<(std::ostream& out, const PolyFisheyeCamera::Parameters& params)
{
    out << "Camera Parameters:" << std::endl;
    out << "|    model_type| "
        << "POLYFISHEYE" << std::endl;
    out << "|   camera_name| " << params.m_cameraName << std::endl;
    out << "|   image_width| " << params.m_imageWidth << std::endl;
    out << "|  image_height| " << params.m_imageHeight << std::endl;

    // projection:  k2, k3, k4, k5, k6, k7, A11, A22, u0, v0
    out << "Projection Parameters" << std::endl;
    out << "polynomial model is:" << std::endl;
    out << " r = "
        << "x + ";
    out << params.m_k2 << " x^2+ ";
    out << params.m_k3 << " x^3+ ";
    out << params.m_k4 << " x^4+ ";
    out << params.m_k5 << " x^5+ ";
    out << params.m_k6 << " x^6+ ";
    out << params.m_k7 << " x^7" << std::endl;

    out << "|            k2| " << params.m_k2 << std::endl;
    out << "|            k3| " << params.m_k3 << std::endl;
    out << "|            k4| " << params.m_k4 << std::endl;
    out << "|            k5| " << params.m_k5 << std::endl;
    out << "|            k6| " << params.m_k6 << std::endl;
    out << "|            k7| " << params.m_k7 << std::endl;
    out << "|           A11| " << params.m_A11 << std::endl;
    out << "|           A12| " << params.m_A12 << std::endl;
    out << "|           A22| " << params.m_A22 << std::endl;
    out << "|            u0| " << params.m_u0 << std::endl;
    out << "|            v0| " << params.m_v0 << std::endl;
    out << "|        isFast| " << params.m_isFast << std::endl;
    out << "|     max_theta| " << params.m_maxIncidentAngle << std::endl;

    return out;
}

void
PolyFisheyeCamera::FastCalcTABLE::setMaxIncidentAngle(double value)
{
    maxIncidentAngle = value;
    resetFastCalc();
}

void
PolyFisheyeCamera::FastCalcTABLE::setNumDiffAngle(int value)
{
    numDiffAngle = value;
}

void
PolyFisheyeCamera::FastCalcTABLE::setMaxImageR(double value)
{
    maxImageR = value;
}

void
PolyFisheyeCamera::FastCalcTABLE::setNumDiffR(int value)
{
    numDiffR = value;
}

eigen_utils::Matrix
PolyFisheyeCamera::FastCalcTABLE::getMatAngleToR()
{
    return angleToR;
}

eigen_utils::Matrix
PolyFisheyeCamera::FastCalcTABLE::getMatRToAngle()
{
    return rToAngle;
}

double
PolyFisheyeCamera::FastCalcTABLE::getMaxIncidentAngle()
{
    return maxIncidentAngle;
}

int
PolyFisheyeCamera::FastCalcTABLE::getNumDiff()
{
    return numDiffR;
}

double
PolyFisheyeCamera::FastCalcTABLE::getDiffAngle()
{
    return diffAngle;
}

double
PolyFisheyeCamera::FastCalcTABLE::getDiffR()
{
    return diffR;
}

bool
PolyFisheyeCamera::FastCalcTABLE::calcAngleToR(eigen_utils::Matrix& _angleToR,
                                               const int    _numDiffAngle,
                                               const double _diffAngle)
{
    std::cout << "poly_angle " << fastPoly->getPolyCoeff().transpose()
              << std::endl;

    _angleToR.resize(_numDiffAngle + 1, 2);

    _angleToR(0, 0) = 0;
    _angleToR(0, 1) = 0;
    for (int index = 1; index <= _numDiffAngle; ++index)
    {
        _angleToR(index, 0) = (double) index * _diffAngle;
        _angleToR(index, 1) = fastPoly->getValue((double) index * _diffAngle);

        if (_angleToR(index, 1) < _angleToR(index - 1, 1))
        {
            std::cout << "#ERROR: polynomial NOT monotone in field!"
                      << std::endl;
            std::cout << "#ERROR: Projection polynomial ERROR!" << std::endl;
            std::cout << "#ERROR: NEED to calibration again!" << std::endl;
            return false;
        }
    }
//    std::cout<<"angleToR "<<_angleToR.transpose()<<std::endl;

    return true;
}

bool
PolyFisheyeCamera::FastCalcTABLE::calcRToAngle(eigen_utils::Matrix& _rToAngle,
                                               const int            _numDiffR,
                                               const double         _diffR,
                                               const double         _maxangle)
{
    std::cout << "poly_r " << fastPoly->getPolyCoeff().transpose() << std::endl;

    _rToAngle.resize(_numDiffR + 1, 2);

    _rToAngle(0, 0) = 0;
    _rToAngle(0, 1) = 0;
    for (int index = 1; index <= _numDiffR; ++index)
    {
        _rToAngle(index, 0) = index * _diffR;
        _rToAngle(index, 1) =
          fastPoly->getOneRealRoot(index * _diffR, 0, _maxangle);

        if (_rToAngle(index, 1) < _rToAngle(index - 1, 1))
        {
            std::cout << "#ERROR: polynomial Root NOT monotone in field!"
                      << std::endl;
            return false;
        }
    }
//    std::cout<<"_rToAngle "<<_rToAngle.transpose()<<std::endl;

    return true;
}

PolyFisheyeCamera::FastCalcTABLE::FastCalcTABLE(eigen_utils::Vector& poly_coeff,
                                                int    num_diff_angle,
                                                double max_angle)
{
    numDiffAngle     = num_diff_angle;
    maxIncidentAngle = max_angle / RAD2DEG;

    fastPoly = new math_utils::Polynomial(poly_coeff);

    std::cout << "fast_poly " << fastPoly->getPolyCoeff().transpose()
              << std::endl;

    resetFastCalc();
}

void
PolyFisheyeCamera::FastCalcTABLE::backprojectSymmetric(
  const Eigen::Vector2d& p_u, double& theta, double& phi) const
{
    double r = p_u.norm();
    //  std::cout << "#INFO: r is " << r << std::endl;

    if (r < 1e-10)
        phi = 0.0;
    else
        phi = atan2(p_u(1), p_u(0));
    //  std::cout << "#INFO: phi is " << phi << std::endl;

    double num = r / diffR;
    int num_down = std::floor(num);
    int num_up = std::ceil(num);

    if (num >= numDiffR || num_up >= numDiffR)
    {
        theta = rToAngle(numDiffR, 1);
        return;
    }
    else
    {
        double theta_up = rToAngle(num_up, 1);
        double theta_down = rToAngle(num_down, 1);

        // linearlize the line with more than 1000 segment
        theta = theta_down + (num - (double)num_down)*(theta_up - theta_down)/(num_up - num_down);
//        std::cout << "theta "<< theta_down<< " " <<theta_up << " " <<theta <<std::endl;
    }
}

double
PolyFisheyeCamera::FastCalcTABLE::r(const double theta) const
{
    if (theta > 1e-10 && theta < maxIncidentAngle)
    {
        double num = theta / diffAngle;
        int num_down = std::floor(num);
        int num_up = std::ceil(num);

        if (num >= numDiffAngle || num_up >= numDiffAngle)
        {
            return angleToR(numDiffAngle, 1);
        }
        else
        {
            double r_up = angleToR(num_up, 1);
            double r_down = angleToR(num_down, 1);

            // linearlize the line with more than 1000 segment
            return ( r_down + (num - (double)num_down)*(r_up - r_down)/(num_up - num_down) );
    //        std::cout << "theta "<< theta_down<< " " <<theta_up << " " <<theta <<std::endl;
        }
    }
    else
        return 0.0;
}

void
PolyFisheyeCamera::FastCalcTABLE::resetFastCalc()
{
    diffAngle = maxIncidentAngle / numDiffAngle;
    calcAngleToR(angleToR, numDiffAngle, diffAngle);

    //      std::cout << "angleToR "<< angleToR <<std::endl;
    //      std::cout << "diffAngle "<< diffAngle*57.29 << " degree"
    //      <<std::endl;

    numDiffR  = numDiffAngle;
    maxImageR = angleToR(numDiffAngle, 1);
    diffR     = maxImageR / numDiffR;
    calcRToAngle(rToAngle, numDiffR, diffR, angleToR(numDiffAngle, 0));
    //    std::cout << "rToAngle "<< rToAngle <<std::endl;
}

PolyFisheyeCamera::FastCalcPOLY::FastCalcPOLY(eigen_utils::Vector& poly_coeff,
                                              double               max_angle)
{
    maxIncidentAngle = max_angle / RAD2DEG;

    fastPoly     = new math_utils::Polynomial(poly_coeff);
    fastRootPoly = new math_utils::Polynomial(ROOT_POLYNOMIAL_ORDER);

    std::cout << "fast_poly " << fastPoly->getPolyCoeff().transpose()
              << std::endl;

    resetFastCalc();
    std::cout << "fastRootPoly " << fastRootPoly->getPolyCoeff().transpose()
              << std::endl;
}

void
PolyFisheyeCamera::FastCalcPOLY::backprojectSymmetric(
  const Eigen::Vector2d& p_u, double& theta, double& phi) const
{

    double r = p_u.norm();
    //  std::cout << "#INFO: r is " << r << std::endl;

    if (r < 1e-10)
        phi = 0.0;
    else
        phi = atan2(p_u(1), p_u(0));
    //  std::cout << "#INFO: phi is " << phi << std::endl;

    theta = this->fastRootPoly->getValue(r);
    //  std::cout << "#INFO: theta is " << theta << std::endl;
}

double
PolyFisheyeCamera::FastCalcPOLY::r(const double theta) const
{
    if (theta > 1e-10 && theta < maxIncidentAngle)
    {
        return fastPoly->getValue(theta);
    }
    else
        return 0.0;
}

void
PolyFisheyeCamera::FastCalcPOLY::setMaxIncidentAngle(double value)
{
    maxIncidentAngle = value;
    resetFastCalc();
}

void
PolyFisheyeCamera::FastCalcPOLY::setMaxImageR(double value)
{
    maxImageR = value;
    resetFastCalc();
}

void
PolyFisheyeCamera::FastCalcPOLY::resetFastCalc()
{
    int    numDiffAngle = 1000;
    double diffAngle    = maxIncidentAngle / numDiffAngle;

    eigen_utils::Vector thetas(numDiffAngle + 1);
    for (int index = 0; index <= numDiffAngle; ++index)
       thetas(index) = diffAngle * index;
    std::cout << "thetas " << thetas.transpose()      << std::endl;

    eigen_utils::Vector rs(numDiffAngle + 1);
    rs = fastPoly->getValue(thetas);
    std::cout << "rs " << rs.transpose()      << std::endl;

    math_utils::PolynomialFit rootPolyFit(7,rs,thetas);
    math_utils::Polynomial poly =  rootPolyFit.getCoeff();

    std::cout << "fastRootPolyIN " << poly.getPolyCoeff().transpose()
              << std::endl;

    *fastRootPoly = rootPolyFit.getPolynomial();
}
}
