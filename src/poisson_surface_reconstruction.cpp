#include "poisson_surface_reconstruction.h"
#include <igl/copyleft/marching_cubes.h>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "fd_interpolate.h"
#include "fd_grad.h"
#include <igl/cat.h>

void poisson_surface_reconstruction(
    const Eigen::MatrixXd & P,
    const Eigen::MatrixXd & N,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & F)
{
  ////////////////////////////////////////////////////////////////////////////
  // Construct FD grid, CONGRATULATIONS! You get this for free!
  ////////////////////////////////////////////////////////////////////////////
  // number of input points
  const int n = P.rows();
  // Grid dimensions
  int nx, ny, nz;
  // Maximum extent (side length of bounding box) of points
  double max_extent =
    (P.colwise().maxCoeff()-P.colwise().minCoeff()).maxCoeff();
  // padding: number of cells beyond bounding box of input points
  const double pad = 8;
  // choose grid spacing (h) so that shortest side gets 30+2*pad samples
  double h  = max_extent/double(30+2*pad);
  // Place bottom-left-front corner of grid at minimum of points minus padding
  Eigen::RowVector3d corner = P.colwise().minCoeff().array()-pad*h;
  // Grid dimensions should be at least 3 
  nx = std::max((P.col(0).maxCoeff()-P.col(0).minCoeff()+(2.*pad)*h)/h,3.);
  ny = std::max((P.col(1).maxCoeff()-P.col(1).minCoeff()+(2.*pad)*h)/h,3.);
  nz = std::max((P.col(2).maxCoeff()-P.col(2).minCoeff()+(2.*pad)*h)/h,3.);
  // Compute positions of grid nodes
  Eigen::MatrixXd x(nx*ny*nz, 3);
  for(int i = 0; i < nx; i++) 
  {
    for(int j = 0; j < ny; j++)
    {
      for(int k = 0; k < nz; k++)
      {
         // Convert subscript to index
         const auto ind = i + nx*(j + k * ny);
         x.row(ind) = corner + h*Eigen::RowVector3d(i,j,k);
      }
    }
  }
  Eigen::VectorXd g = Eigen::VectorXd::Zero(nx*ny*nz);

  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  Eigen::SparseMatrix<double> Wx;
  Eigen::RowVector3d corner_x;
  corner_x(0) = corner(0) + h/2;
  corner_x(1) = corner(1);
  corner_x(2) = corner(2);
  fd_interpolate(nx-1, ny, nz, h, corner_x, P, Wx);
  Eigen::MatrixXd Vx = Wx.transpose() * N.col(0);

  Eigen::SparseMatrix<double> Wy;
  Eigen::RowVector3d corner_y;
  corner_y(0) = corner(0);
  corner_y(1) = corner(1) + h/2;
  corner_y(2) = corner(2);
  fd_interpolate(nx, ny-1, nz, h, corner_y, P, Wy);
  Eigen::MatrixXd Vy = Wy.transpose() * N.col(1);

  Eigen::SparseMatrix<double> Wz;
  Eigen::RowVector3d corner_z;
  corner_z(0) = corner(0);
  corner_z(1) = corner(1);
  corner_z(2) = corner(2) + h/2;
  fd_interpolate(nx, ny, nz-1, h, corner_z, P, Wz);
  Eigen::MatrixXd Vz = Wz.transpose() * N.col(2);

  Eigen::MatrixXd v(Vx.rows()+Vy.rows()+Vz.rows(), Vx.cols());
  v << Vx, Vy, Vz; 

  Eigen::SparseMatrix<double> G;
  fd_grad(nx, ny, nz, h, G);

  Eigen::BiCGSTAB<Eigen::SparseMatrix<double> > solver;
  solver.compute(G.transpose() * G);
  g = solver.solve(G.transpose() * v);

  std::cout << "#iterations:     " << solver.iterations() << std::endl;
  std::cout << "estimated error: " << solver.error()      << std::endl;

  // std::cout<<"g.row:"<<g.rows()<<", g.col:"<<g.cols()<<", g:"<<g<<std::endl;
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Run black box algorithm to compute mesh from implicit function: this
  // function always extracts g=0, so "pre-shift" your g values by -sigma
  ////////////////////////////////////////////////////////////////////////////
  igl::copyleft::marching_cubes(g, x, nx, ny, nz, V, F);
}
