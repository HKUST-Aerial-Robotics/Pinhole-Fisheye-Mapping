#include <camera_model/code_utils/eigen_utils.h>
#include <camera_model/camera_models/PolyFisheyeCamera.h>

namespace camera_model
{

class PolyFisheyeCUDA
{
public:
    PolyFisheyeCUDA(std::string camera_model_file)
    {
        loadCameraFile(camera_model_file);
        cudaDataTrans();
    }

    void loadCameraFile(std::string camera_model_file)
    {
        camera_model::PolyFisheyeCameraPtr camera(new camera_model::PolyFisheyeCamera);

        camera_model::PolyFisheyeCamera::Parameters params = camera->getParameters();
        params.readFromYamlFile(camera_model_file);
        camera->setParameters(params);

        std::cout<<"camera_model_file " << camera_model_file<<std::endl;
        // project parameters
        A11 = camera->getParameters().A11();
        A12 = camera->getParameters().A12();
        A22 = camera->getParameters().A22();
        u0 = camera->getParameters().u0();
        v0 = camera->getParameters().v0();
        std::cout<<"A11 " << A11 <<std::endl;
        std::cout<<"A12 " << A12 <<std::endl;
        std::cout<<"A22 " << A22 <<std::endl;
        std::cout<<"u0  " << u0  <<std::endl;
        std::cout<<"v0  " << v0  <<std::endl;

        // inverse project parameters
        m_inv_K11 = camera->getInv_K11();
        m_inv_K12 = camera->getInv_K12();
        m_inv_K13 = camera->getInv_K13();
        m_inv_K22 = camera->getInv_K22();
        m_inv_K23 = camera->getInv_K23();
        std::cout<<"m_inv_K11 " << m_inv_K11 <<std::endl;
        std::cout<<"m_inv_K12 " << m_inv_K12 <<std::endl;
        std::cout<<"m_inv_K13 " << m_inv_K13 <<std::endl;
        std::cout<<"m_inv_K22 " << m_inv_K22 <<std::endl;
        std::cout<<"m_inv_K23 " << m_inv_K23 <<std::endl;

        // fast calculate tables and parameters
        numDiff = camera->getFastCalc()->getNumDiff();
        diffR   = camera->getFastCalc()->getDiffR();
        diffAngle = camera->getFastCalc()->getDiffAngle();
        maxIncidentAngle = camera->getFastCalc()->getMaxIncidentAngle();
        std::cout<<"numDiff   " << numDiff   <<std::endl;
        std::cout<<"diffR     " << diffR     <<std::endl;
        std::cout<<"diffAngle " << diffAngle <<std::endl;
        std::cout<<"maxIncidentAngle " << maxIncidentAngle <<std::endl;

        angleToR = camera->getFastCalc()->getMatAngleToR();
        rToAngle = camera->getFastCalc()->getMatRToAngle();
        std::cout<<"m_inv_K22 " << m_inv_K22 <<std::endl;
        std::cout<<"m_inv_K23 " << m_inv_K23 <<std::endl;
    }

    void cudaDataTrans()
    {
        //TODO: trans data to GPU
    }


    void
    spaceToPlane(const Eigen::Vector3d& P,
                 Eigen::Vector2d&       p) const
    {
        double theta = acos(P(2) / P.norm());
        double phi   = atan2(P(1), P(0));

        // TODO
        Eigen::Vector2d p_u;
            p_u = r(theta) * Eigen::Vector2d(cos(phi), sin(phi));
            //    std::cout<< " p_u " << p_u.transpose()<<std::endl;

        // Apply generalised projection matrix
        p(0) = A11 * p_u(0) + A12 * p_u(1) +
               u0;
        p(1) = A22 * p_u(1) + v0;
    }

    void
    liftProjective(const Eigen::Vector2d& p,
                   Eigen::Vector3d&       P) const
    {
        // Lift points to normalised plane
        double theta, phi;
        // Obtain a projective ray

        backprojectSymmetric(
                    Eigen::Vector2d(m_inv_K11 * p(0) + m_inv_K12 * p(1) + m_inv_K13,
                                    m_inv_K22 * p(1) + m_inv_K23),
                    theta, phi);

        P =
                Eigen::Vector3d(cos(phi) * sin(theta),
                                sin(phi) * sin(theta),
                                cos(theta));
    }

    double
    r(const double theta) const
    {
        if (theta > 1e-10 && theta < maxIncidentAngle)
        {
            double num = theta / diffAngle;
            int num_down = std::floor(num);
            int num_up = std::ceil(num);

            if (num >= numDiff || num_up >= numDiff)
            {
                return angleToR(numDiff, 1);
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
    backprojectSymmetric(
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

        if (num >= numDiff || num_up >= numDiff)
        {
            theta = rToAngle(numDiff, 1);
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


public:
    double A11, A12, A22, u0, v0;
    double m_inv_K11, m_inv_K12, m_inv_K13, m_inv_K22, m_inv_K23;

    double diffR, diffAngle, maxIncidentAngle;
    int numDiff;
    eigen_utils::Matrix angleToR;
    eigen_utils::Matrix rToAngle;
};


}
