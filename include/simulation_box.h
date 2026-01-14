#ifndef __LIBHERIC_SIMULATION_BOX_H__
#define __LIBHERIC_SIMULATION_BOX_H__

#include "constants.h"
#include <Eigen/Dense>

namespace libheric
{
    struct simulation_box 
    {
        simulation_box(const Eigen::Matrix3d&  cell_mat, 
                       const Eigen::Matrix3Xd& positions,
                       const Eigen::VectorXi&  kinds,
                       const Eigen::VectorXd&  exponents,
                       const Eigen::MatrixXd&  coefficients,
                       const Eigen::VectorXi&  kind_offsets,
                       const Eigen::VectorXi&  kind_sizes,
                       const Eigen::MatrixXi&  set_offsets,
                       const Eigen::MatrixXi&  set_sizes,
                       const Eigen::Vector<int, MAX_AM + 1>& shell_offsets,
                       const Eigen::Vector<int, MAX_AM + 1>& shell_sizes) 
            : 
            cell_mat     (cell_mat     ),
            positions    (positions    ),
            kinds        (kinds        ),
            exponents    (exponents    ),
            coefficients (coefficients ),
            kind_offsets (kind_offsets ),
            kind_sizes   (kind_sizes   ),
            set_offsets  (set_offsets  ),
            set_sizes    (set_sizes    ),
            shell_offsets(shell_offsets),
            shell_sizes  (shell_sizes  )
            {}

        Eigen::Matrix3d  cell_mat; 
        Eigen::Matrix3Xd positions;
        Eigen::VectorXi  kinds;
        Eigen::VectorXd  exponents;
        Eigen::MatrixXd  coefficients;
        Eigen::VectorXi  kind_offsets;
        Eigen::VectorXi  kind_sizes;
        Eigen::MatrixXi  set_offsets;
        Eigen::MatrixXi  set_sizes;
        Eigen::Vector<int, MAX_AM + 1> shell_offsets;
        Eigen::Vector<int, MAX_AM + 1> shell_sizes;

    };
}
#endif
