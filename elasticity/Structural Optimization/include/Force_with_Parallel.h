#pragma once
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <stdlib.h>
#include <fstream>
#include <iostream>
#include "Deal_II_MMA.h"
// 
using namespace dealii;
const int dim = 2;
const double epsimin = 1e-3;
const unsigned int degree = 1;
class ForceProblem
{
public:
    ForceProblem();
    void run();
private:
    // 位移场
    void make_grid();
    void setup_system();
    void assemble_system();
    void assemble_system_rhs();
    void solve();
    void get_object_diff_values(LA::MPI::Vector &Object_Diff_Values);
    void set_cell_material_id(); 
    void set_cell_nothing_values(LA::MPI::Vector &Diff_Values);
    // 过滤场
    void setup_filter_system();
    void assemble_filter_system();
    void assemble_filter_rhs(LA::MPI::Vector Vec);
    void solve_filter();
    void generate_average_vector(LA::MPI::Vector &out_average);
    // 密度场
    void init_simp(); 
    void get_cell_volume();
    void output_results(unsigned int loop);
    void Refine_Grid();
    void time();    
    // 
    MPI_Comm mpi_communicator;
    const unsigned int this_mpi_process;
    parallel::distributed::Triangulation<dim> triangulation;
    // 
    hp::MappingCollection<dim> mapping_fe;    
    hp::MappingCollection<dim> mapping_fe_filter;
    //    
    hp::FECollection<dim>      fe;
    hp::FECollection<dim>      fe_rho;
    hp::FECollection<dim>      fe_filter;  
    //   
    hp::QCollection<dim>       quadrature_formula;
    hp::QCollection<dim-1>     quadrature_formula_face;
    hp::QCollection<dim>       quadrature_formula_rho; 
    hp::QCollection<dim>       quadrature_formula_filter;   
    // 
    DoFHandler<dim> dof_handler;
    DoFHandler<dim> dof_handler_rho;
    DoFHandler<dim> dof_handler_filter;
    // 
    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    // 
    IndexSet locally_owned_dofs_rho;
    // 
    IndexSet locally_owned_dofs_filter;
    IndexSet locally_relevant_dofs_filter; 
    // 
    AffineConstraints<double> constraints;
    AffineConstraints<double> constraints_filter;
    // 
    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;
    LA::MPI::Vector       completely_distributed_solution;
    // 
    LA::MPI::SparseMatrix system_matrix_filter;
    LA::MPI::Vector       locally_relevant_solution_filter;
    LA::MPI::Vector       system_rhs_filter;
    LA::MPI::Vector       completely_distributed_solution_f;
    // 
    ConditionalOStream    pcout;
    TimerOutput           computing_timer;
    // 密度变量
    LA::MPI::Vector    Simp_Rho;
    LA::MPI::Vector    Simp_Rho_Filted;
    LA::MPI::Vector    Simp_Max;
    LA::MPI::Vector    Simp_Min;
    LA::MPI::Vector    Cell_Volume;
    LA::MPI::Vector    Object_Diff_Values;
    std::vector<LA::MPI::Vector> Constraint_Diff_Values;    
    // 
    Deal_II_MMA_Solver               MMA_Solver;
    // 
    double possion = 0.3;
    double E_all = 2e11;
    double penal = 4;
    double volfrac = 0.3;
    double Converage_Epsi = 1e-3;
    unsigned int init_refine_times = 8;
    int Loop_Max = 200;    
};