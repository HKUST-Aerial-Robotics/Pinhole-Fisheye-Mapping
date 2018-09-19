#ifndef POLYFISHEYECAMERA_H
#define POLYFISHEYECAMERA_H

#include <opencv2/core/core.hpp>
#include <string>

#include "ceres/rotation.h"

#include "Camera.h"

#include <camera_model/code_utils/math_utils/Polynomial.h>

#define MAX_INCIDENT_ANGLE_DEGREE 130

// k2 k3 k4 k5 k6 k7 A11 A22 u0 v0
#define FISHEYE_POLY_ORDER 7
#define FISHEYE_AFFINE_NUM 3
#define FISHEYE_TANGENT_NUM 4
#define FISHEYE_PARAMS_NUM (FISHEYE_POLY_ORDER - 1 + FISHEYE_AFFINE_NUM + 2)

#define FAST_NUM_DEFAULT 3000
#define FAST_MAX_INCIDENT_ANGLE 100

namespace camera_model
{

class PolyFisheyeCamera : public Camera
{
  public:
    class Parameters : public Camera::Parameters
    {
      public:
        Parameters();
        Parameters(const std::string& cameraName, int w, int h, double k2,
                   double k3, double k4, double k5, double k6, double k7,
                   double A11, double A12, double A22, double u0, double v0,
                   int isFast);

        double& k2(void);
        double& k3(void);
        double& k4(void);
        double& k5(void);
        double& k6(void);
        double& k7(void);
        double& A11(void);
        double& A12(void);
        double& A22(void);
        double& u0(void);
        double& v0(void);

        double k2(void) const;
        double k3(void) const;
        double k4(void) const;
        double k5(void) const;
        double k6(void) const;
        double k7(void) const;
        double A11(void) const;
        double A12(void) const;
        double A22(void) const;
        double u0(void) const;
        double v0(void) const;

        int& isFast(void);
        int& numDiff(void);
        int& maxIncidentAngle(void);
        int  isFast(void) const;
        int  numDiff(void) const;
        int  maxIncidentAngle(void) const;

        bool readFromYamlFile(const std::string& filename);

        void writeToYamlFile(const std::string& filename) const;

        Parameters& operator=(const Parameters& other);

        friend std::ostream& operator<<(std::ostream&     out,
                                        const Parameters& params);

      private:
        // projection parameters
        double m_k2;
        double m_k3;
        double m_k4;
        double m_k5;
        double m_k6;
        double m_k7;

        double m_A11;
        double m_A12;
        double m_A22;
        double m_u0;
        double m_v0;

      private:
        int m_isFast; // 0:disable, 1:able
        int m_numDiff;
        int m_maxIncidentAngle;
    };

    class FastCalcPOLY
    {
      #define ROOT_POLYNOMIAL_ORDER (2*FISHEYE_POLY_ORDER)
      public:
        FastCalcPOLY(eigen_utils::Vector& poly_coeff,
                     double max_angle);

        void backprojectSymmetric(const Eigen::Vector2d& p_u, double& theta,
                                  double& phi) const;
        double r(const double theta) const;

        void setMaxIncidentAngle(double value);
        void setMaxImageR(double value);

      private:
        void resetFastCalc();

      private:
        math_utils::Polynomial* fastPoly;
        math_utils::Polynomial* fastRootPoly;

        double maxIncidentAngle;
        double maxImageR;
    };

    class FastCalcTABLE
    {
      public:
        FastCalcTABLE(eigen_utils::Vector& poly_coeff, int num_diff_angle,
                      double max_angle);

        void backprojectSymmetric(const Eigen::Vector2d& p_u, double& theta,
                                  double& phi) const;
        double r(const double theta) const;

        void setMaxIncidentAngle(double value);
        void setNumDiffAngle(int value);
        void setMaxImageR(double value);
        void setNumDiffR(int value);

        eigen_utils::Matrix getMatAngleToR();
        eigen_utils::Matrix getMatRToAngle();

        double getMaxIncidentAngle();
        int    getNumDiff();
        double getDiffAngle();
        double getDiffR();

      private:
        void resetFastCalc();
        bool calcAngleToR(eigen_utils::Matrix& _angleToR,
                          const int _numDiffAngle, const double _diffAngle);
        bool calcRToAngle(eigen_utils::Matrix& _rToAngle, const int _numDiffR,
                          const double _diffR, const double _maxangle);

      private:
        math_utils::Polynomial* fastPoly;

        double maxIncidentAngle;
        double maxImageR;
        int    numDiffAngle;
        int    numDiffR;
        double diffAngle;
        double diffR;

        eigen_utils::Matrix angleToR;
        eigen_utils::Matrix rToAngle;
    };

    PolyFisheyeCamera();

    PolyFisheyeCamera(const std::string& cameraName, int imageWidth,
                      int imageHeight, double k2, double k3, double k4,
                      double k5, double k6, double k7, double A11, double A12,
                      double A22, double u0, double v0, int isFast);

    PolyFisheyeCamera(const Parameters& params);


    Camera::ModelType  modelType(void) const;
    const std::string& cameraName(void) const;

    int imageWidth(void) const;
    int imageHeight(void) const;

    double get_theta(int x, int y);
    double getMaxIncidentAngle();
    double get_diffangle() const;
    double get_diffr() const;    
    eigen_utils::Matrix getMatAngleToR();
    eigen_utils::Matrix getMatRToAngle();

    void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const;

    void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                      Eigen::Matrix<double, 2, 3>& J) const;

    void estimateIntrinsics(const cv::Size& boardSize,
      const std::vector<std::vector<cv::Point3f>>& objectPoints,
      const std::vector<std::vector<cv::Point2f>>& imagePoints);

    virtual void liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;

    virtual void rayToPlane(const Ray& ray, Eigen::Vector2d& p) const;

    // Lift points from the image plane to the projective space
    virtual void liftProjective(const Eigen::Vector2d& p,
                                Eigen::Vector3d&       P) const;

    virtual void liftProjectiveToRay(const Eigen::Vector2d& p, Ray& ray) const;

    virtual void undistToPlane(const Eigen::Vector2d& p_u,
                               Eigen::Vector2d&       p) const;

    template <typename T>
    static void spaceToPlane(const T* const params, const T* const q,
                             const T* const t, const Eigen::Matrix<T, 3, 1>& P,
                             Eigen::Matrix<T, 2, 1>& p);

    // virtual void initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale
    // =
    // 1.0) const = 0;
    virtual cv::Mat initUndistortRectifyMap(
      cv::Mat& map1, cv::Mat& map2, float fx = -1.0f, float fy = -1.0f,
      cv::Size imageSize = cv::Size(0, 0), float cx = -1.0f, float cy = -1.0f,
      cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const;

    int parameterCount(void) const;

    const Parameters& getParameters(void) const;

    void setParameters(const Parameters& parameters);

    void readParameters(const std::vector<double>& parameterVec);

    void writeParameters(std::vector<double>& parameterVec) const;

    void writeParametersToYamlFile(const std::string& filename) const;

    std::string parametersToString(void) const;

    math_utils::Polynomial* getPoly() const;
    void setPoly(math_utils::Polynomial* value);

    FastCalcTABLE* getFastCalc();

    double getInv_K11() const;
    double getInv_K12() const;
    double getInv_K13() const;
    double getInv_K22() const;
    double getInv_K23() const;

  private:
    template <typename T>
    static T r(T k2, T k3, T k4, T k5, T k6, T k7, T theta);

    void backprojectSymmetric(const Eigen::Vector2d& p_u, double& theta,
                              double& phi) const;
    bool calcKinvese(double a11, double a12, double a22, double u0, double v0);

    Parameters mParameters;

  protected:
    math_utils::Polynomial* poly;
//    FastCalcPOLY*           fastCalc;
    FastCalcTABLE*          fastCalc;

    double m_inv_K11, m_inv_K12, m_inv_K13, m_inv_K22, m_inv_K23;
};

typedef boost::shared_ptr<PolyFisheyeCamera>       PolyFisheyeCameraPtr;
typedef boost::shared_ptr<const PolyFisheyeCamera> PolyFisheyeCameraConstPtr;

template <typename T>
T
PolyFisheyeCamera::r(T k2, T k3, T k4, T k5, T k6, T k7, T theta)
{
    return theta +
      k2 * theta * theta +
      k3 * theta * theta * theta +
      k4 * theta * theta * theta * theta +
      k5 * theta * theta * theta * theta * theta +
      k6 * theta * theta * theta * theta * theta * theta +
      k7 * theta * theta * theta * theta * theta * theta * theta;
}

template <typename T>
void
PolyFisheyeCamera::spaceToPlane(const T* const params, const T* const q,
                                const T* const t,
                                const Eigen::Matrix<T, 3, 1>& P,
                                Eigen::Matrix<T, 2, 1>&       p)
{
    T P_w[3];
    P_w[0] = T(P(0));
    P_w[1] = T(P(1));
    P_w[2] = T(P(2));

    // Eigen convention (x, y, z, w)
    // Ceres convention (w, x, y, z)
    T q_ceres[4] = { q[3], q[0], q[1], q[2] };

    T P_c[3];
    ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

    P_c[0] += t[0];
    P_c[1] += t[1];
    P_c[2] += t[2];

    // project 3D object point to the image plane;
    T k2  = params[0];
    T k3  = params[1];
    T k4  = params[2];
    T k5  = params[3];
    T k6  = params[4];
    T k7  = params[5];
    T A11 = params[6];
    T A12 = params[7];
    T A22 = params[8];
    T u0  = params[9];
    T v0  = params[10];

    T len   = sqrt(P_c[0] * P_c[0] + P_c[1] * P_c[1] + P_c[2] * P_c[2]);
    T theta = acos(P_c[2] / len);
    T phi   = atan2(P_c[1], P_c[0]);

    Eigen::Matrix<T, 2, 1> p_u = r(k2, k3, k4, k5, k6, k7, theta) *
                                 Eigen::Matrix<T, 2, 1>(cos(phi), sin(phi));

    p(0) = A11 * p_u(0) + A12 * p_u(1) + u0;
    p(1) =                A22 * p_u(1) + v0;
}
}
#endif // POLYFISHEYECAMERA_H
