#include "fd_interpolate.h"
#include <iostream>

void fd_interpolate(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const Eigen::RowVector3d & corner,
  const Eigen::MatrixXd & P,
  Eigen::SparseMatrix<double> & W)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  W.resize(P.rows(), nx*ny*nz);
  std::vector<Eigen::Triplet<double>> tripletList;
  for (int m = 0; m < P.rows(); ++m)
  {
    int i = (P(m,0) - corner[0]) / h;
    int j = (P(m,1) - corner[1]) / h;
    int k = (P(m,2) - corner[2]) / h;

    double x_d = (P(m,0) - corner[0] - i*h) / h;
    double y_d = (P(m,1) - corner[1] - j*h) / h;
    double z_d = (P(m,2) - corner[2] - k*h) / h;

    tripletList.push_back(Eigen::Triplet<double>(m, i+j*nx+k*nx*ny, (1-x_d)*(1-y_d)*(1-z_d)));       //000
    tripletList.push_back(Eigen::Triplet<double>(m, (i+1)+j*nx+k*nx*ny, x_d*(1-y_d)*(1-z_d)));       //100
    tripletList.push_back(Eigen::Triplet<double>(m, i+j*nx+(k+1)*nx*ny, (1-x_d)*(1-y_d)*z_d));       //001
    tripletList.push_back(Eigen::Triplet<double>(m, (i+1)+j*nx+(k+1)*nx*ny, x_d*(1-y_d)*z_d));       //101
    tripletList.push_back(Eigen::Triplet<double>(m, i+(j+1)*nx+k*nx*ny, (1-x_d)*y_d*(1-z_d)));       //010
    tripletList.push_back(Eigen::Triplet<double>(m, (i+1)+(j+1)*nx+k*nx*ny, x_d*y_d*(1-z_d)));       //110
    tripletList.push_back(Eigen::Triplet<double>(m, i+(j+1)*nx+(k+1)*nx*ny, (1-x_d)*y_d*z_d));       //011
    tripletList.push_back(Eigen::Triplet<double>(m, (i+1)+(j+1)*nx+(k+1)*nx*ny, x_d*y_d*z_d));       //111
  }
  W.setFromTriplets(tripletList.begin(), tripletList.end());
  ////////////////////////////////////////////////////////////////////////////
}
