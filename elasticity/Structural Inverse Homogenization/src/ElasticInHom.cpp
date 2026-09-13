#include "../include/Elastic_In_Hom.h"

int main(int argc, char *argv[])
{
    try 
    {
        using namespace dealii;
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        ElasticHomogenization problem;
        problem.Run();
    }  
    catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
 
      return 1;
    }
    return 0;
};