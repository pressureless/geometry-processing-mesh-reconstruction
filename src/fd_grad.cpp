#include "fd_grad.h"
#include "fd_partial_derivative.h"
#include <iostream>

void fd_grad(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  Eigen::SparseMatrix<double> & G)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
	// G.resize((nx-1)*ny*nz + nx*(ny-1)*nz + nx*ny*(nz-1), nx*ny*nz);
	Eigen::SparseMatrix<double> Gx, Gy, Gz;
	fd_partial_derivative(nx, ny, nz, h, 0, Gx);
	fd_partial_derivative(nx, ny, nz, h, 1, Gy);
	fd_partial_derivative(nx, ny, nz, h, 2, Gz);

	G.resize(Gx.rows() + Gy.rows() + Gz.rows(), Gx.cols());
	G.reserve(Gx.nonZeros() + Gy.nonZeros()+ Gz.nonZeros());
	for(int c=0; c<Gx.cols(); ++c)
	{
	    G.startVec(c); 
	    for(Eigen::SparseMatrix<double>::InnerIterator itL(Gx, c); itL; ++itL)
	         G.insertBack(itL.row(), c) = itL.value();
	    for(Eigen::SparseMatrix<double>::InnerIterator itC(Gy, c); itC; ++itC)
	         G.insertBack(itC.row()+Gx.rows(), c) = itC.value();
	    for(Eigen::SparseMatrix<double>::InnerIterator itT(Gz, c); itT; ++itT)
	         G.insertBack(itT.row()+Gx.rows()+Gy.rows(), c) = itT.value();
	}
	G.finalize(); 
  ////////////////////////////////////////////////////////////////////////////
}
