#include "../include/Force_with_Parallel.h"
int main(int argc, char *argv[])
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ForceProblem force_opt;  
  force_opt.run();
  return 0;
}
