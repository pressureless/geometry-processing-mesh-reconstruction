#include "fd_partial_derivative.h"

void fd_partial_derivative(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const int dir,
  Eigen::SparseMatrix<double> & D)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  std::vector<Eigen::Triplet<double>> tripletList;
  if (dir == 0)
  {
  	D.resize((nx-1)*ny*nz, nx*ny*nz);
  	for (int i = 0; i < nx-1; ++i)
  	{
  		for (int j = 0; j < ny; ++j)
  		{
  			for (int k = 0; k < nz; ++k)
  			{
  				int left = i;
  				int right = i+1;
  				tripletList.push_back(Eigen::Triplet<double>(i+j*(nx-1)+k*(nx-1)*ny, left+j*nx+k*nx*ny, -1));   
  				tripletList.push_back(Eigen::Triplet<double>(i+j*(nx-1)+k*(nx-1)*ny, right+j*nx+k*nx*ny, 1));   
  			}
  		}    
  	}
  }
  else if (dir == 1)
  {
    D.resize(nx*(ny-1)*nz, nx*ny*nz);
    for (int i = 0; i < nx; ++i)
  	{
  		for (int j = 0; j < ny-1; ++j)
  		{
  			for (int k = 0; k < nz; ++k)
  			{
  				int left = j;
  				int right = j+1;
  				tripletList.push_back(Eigen::Triplet<double>(i+j*nx+k*nx*(ny-1), i+left*nx+k*nx*ny, -1));   
  				tripletList.push_back(Eigen::Triplet<double>(i+j*nx+k*nx*(ny-1), i+right*nx+k*nx*ny, 1));   
  			}
  		}    
  	}
  }
  else{
  	D.resize(nx*ny*(nz-1), nx*ny*nz);
	for (int i = 0; i < nx; ++i)
	{
	  	for (int j = 0; j < ny; ++j)
	  	{
	  		for (int k = 0; k < nz-1; ++k)
	  		{
	  			int left = k;
	  			int right = k+1;
	  			tripletList.push_back(Eigen::Triplet<double>(i+j*nx+k*nx*ny, i+j*nx+left*nx*ny, -1));   
	  			tripletList.push_back(Eigen::Triplet<double>(i+j*nx+k*nx*ny, i+j*nx+right*nx*ny, 1));   
	  		}
	  	}    
	}
  }
  D.setFromTriplets(tripletList.begin(), tripletList.end());
  ////////////////////////////////////////////////////////////////////////////
}
