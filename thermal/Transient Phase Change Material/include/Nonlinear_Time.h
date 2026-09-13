/*
    \brief This program is a topology optimization program for nonlinear time-discretized problems.
           It mainly includes a nonlinear time-discretization solver,
           a discrete adjoint sensitivity solver,
           a filter,
           and an MMA optimization solver.
    \author zxl
    \date 2025.2.16
*/
#pragma once

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/petsc_ts.h>

#include <fstream>
#include <iostream>

#include "Deal_II_MMA_Parallel.h"

const unsigned int dim = 2;
const unsigned int degree = 1;
using namespace dealii;

class PCM
{
    public :
        // Given the current temperature t and design variable value x, return cp.
        double cp_t_x_value(double t, double x);
        // Given the current temperature t and design variable value x, return dcp_dt.
        double dcp_dt_x_value(double t, double x);
        // Given the current temperature t and design variable value x, return dcp_dx.
        double dcp_t_dx_value(double t, double x);
    private :
        const double delta_t = 8;
        const double cp_s = 100;
        const double cp_l = 100;
        const double l = 1e3;
        const double t_start=303.15;
};

class SIMP
{
    public :
        double simp_value(double x);
        double simp_grad_value(double x);
    private :
        const double penalty = 3.;
};

class OrderSIMP
{
    public :
        double simp_value(double x);
        double simp_grad_value(double x);
    private :
        const double penalty = 3.;
        const std::vector<double> rho = {1e-3, 0.1, 0.8, 1.};
        const std::vector<double> kappa = {1e-3, 0.1, 0.8, 1.};
};

class Nonlinear_Time
{
public:
    Nonlinear_Time();
    void run();

private:
    // Generate the mesh.
    void make_grid();
    // Forward solve.
        /*
            Initialize the heat transfer system.
            Must be called before setup_adjoint_solver().
        */
        void setup_forward_system();
        /*
            \brief Compute the residual vector of the fully implicit time-discretized
                   equation, Res = Mu_dot + Au - Q.
            \param time          Time.
            \param solution      Solution vector.
            \param solution_dot  Time derivative of the solution vector.
            \param residual      Residual vector.
        */
        void implicit_function(const double time,
                               const PETScWrappers::MPI::Vector &solution,
                               const PETScWrappers::MPI::Vector &solution_dot,
                               PETScWrappers::MPI::Vector &residual);
        /*
            \brief Compute the Jacobian of the fully implicit time-discretized
                   equation F with respect to u,
            J = \frac{\partial F(t,u,u_dot)}{\partial u} + \alpha \frac{\partial F(t,u,u_dot)}{\partial u_dot}
                   for linear system
                   F'(t,u,u_dot) = \nabla\psi \kappa \nabla\psi + \alpha \psi \rho cp \psi
                   for nonlinear system
                   F'(t,u,u_dot) = \nabla\psi \kappa(u) \nabla\psi +
                                   \psi \kappa'(u) (\nabla u) \nabla\psi +
                                   \psi \rho cp'(u) u_dot \psi +
                                   \alpha \psi \rho cp \psi
            \param time          Time.
            \param solution      Solution vector.
            \param solution_dot  Time derivative of the solution vector.
            \param alpha         Shift parameter (1/\Delta t).
        */
        void assemble_implicit_jacobian_u(const double time,
                                          const PETScWrappers::MPI::Vector &solution,
                                          const PETScWrappers::MPI::Vector &solution_dot,
                                          const double alpha);
        // /*
        //     \brief Solve the linear system, F'(t,u,u_dot) \delta u = -F(t,u,u_dot).
        //     \param solution_delta  Increment of the solution vector.
        //     \param residual        Residual vector.
        // */
        // void solve_with_jacobian(const PETScWrappers::MPI::Vector &residual,
        //                          PETScWrappers::MPI::Vector &solution_delta);
        /*
            \brief Output the forward solve results.
            \param time             Time.
            \param solution         Solution vector.
            \param timestep_number  Time step number.
        */
        void output_forward_results(const double time,
                                    const unsigned int timestep_number,
                                    const PETScWrappers::MPI::Vector &solution);
        /*
            \brief Set up the forward solver.
        */
        void setup_forward_solver();
    // Backward adjoint solve.
        /*
            Initialize the adjoint system.
            Must be called after setup_forward_solver().
        */
        void setup_adjoint_system();
        /*
            \brief Initialize the initial values of the adjoint vectors.
        */
        void initialize_adjoint_vector();
        /*
            \brief Compute the integral function r,
                   r = \frac{1}{N} \sum_{i=1}^{N} (u_i - \overline{u})^2.
            \param solution  Solution vector.
            \param integral  Integral vector.
        */
        void integral_funtion(const PETScWrappers::MPI::Vector &solution,
                              PETScWrappers::MPI::Vector &integral);
        /*
            \brief Assemble the Jacobian of the implicit function with respect to
                   the design variables,
            F_{p_i} = \nabla\psi \kappa(u,p)_{p_i} \nabla\psi u +
                      \psi \rho cp(u,p)_{p_i} \psi u_dot
            \param time          Time.
            \param solution      Solution vector.
            \param solution_dot  Time derivative of the solution vector.
            \param residual      Residual vector.
        */
        void assemble_implicit_jacobian_p(const double time,
                                          const PETScWrappers::MPI::Vector &solution,
                                          const PETScWrappers::MPI::Vector &solution_dot);
        /*
            \brief Assemble the Jacobian of the integral function r with respect to
                   the state variables,
            r_{u} = 2/N(u - \overline{u})
            \param time          Time.
            \param solution      Solution vector.
            \param solution_dot  Time derivative of the solution vector.
            \param residual      Residual vector.
        */
        void assemble_integral_jacobian_u(const PETScWrappers::MPI::Vector &solution);
        /*
            \brief Assemble the Jacobian of the integral function r with respect to
                   the design variables,
            r_{p} = 0
        */
        void assemble_integral_jacobian_p();
        /*
            \brief Output the adjoint solve results.
            \param time             Time.
            \param timestep_number  Time step number.
            \param solution         Solution vector.
        */
        void output_adjoint_results(const double time,
                                    const unsigned int timestep_number);
        /*
            \brief Set up the adjoint solver.
        */
        void setup_adjoint_solver();
    //  Helmholtz filter equation.
        /*
            \brief Initialize the filter matrix and vectors.
        */
        void setup_filter_system();
        /*
            \brief Assemble the filter equation matrix.
        */
        void assemble_filter_matrix();
        /*
            \brief Assemble the right-hand side of the filter equation.
        */
        void assemble_filter_rhs(PETScWrappers::MPI::Vector intput);
        /*
            \brief Solve the filter equation.
        */
        void solve_filter_system();
        /*
            \brief Generate the filtered vector.
        */
        void generate_average_solution(PETScWrappers::MPI::Vector &output);
        /*
            \brief Wrapper.
        */
        void helmholtz_filter(PETScWrappers::MPI::Vector intput,
                              PETScWrappers::MPI::Vector &output);
    void get_cell_volume();
    void output_sensitivity_optimization(unsigned int iter);
    void time();
    // MPI communication.
    const MPI_Comm mpi_communicator;
    // Mesh.
    parallel::distributed::Triangulation<dim> triangulation;
    // Finite elements.
    FE_Q<dim>   fe;
    FE_Q<dim>   fe_filter;
    FE_DGQ<dim> fe_simp;
    // DoF handling.
    DoFHandler<dim> dof_handler;
    DoFHandler<dim> dof_handler_filter;
    DoFHandler<dim> dof_handler_simp;
    // Forward solver for temperature.
        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;
        /*
            \frac{\partial F(t,u,u_dot)}{\partial u}
        */
        PETScWrappers::MPI::SparseMatrix jacobian_matrix_dFdU;
        /*
            u(t_{N})
        */
        PETScWrappers::MPI::Vector tempture_solution;
        /*
            tmp vectors
        */
        PETScWrappers::MPI::Vector tmp_solution,
                                   tmp_solution_dot;
        PETScWrappers::MPI::Vector locally_relevant_solution,
                                   locally_relevant_solution_dot;

        AffineConstraints<double> current_constraints;
        AffineConstraints<double> homogeneous_constraints;

        PETScWrappers::TimeStepperData time_stepper_data;
        PETScWrappers::TimeStepper<PETScWrappers::MPI::Vector,
                                   PETScWrappers::MPI::SparseMatrix>
                                   time_stepper;
        TS quadts;
        TSTrajectory tj;
        unsigned int initial_global_refinement;
    // Adjoint solver for sensitivity.
        IndexSet locally_owned_dofs_simp;
        IndexSet locally_relevant_dofs_simp;
        /*
            \frac{\partial F(t,u;P,u_dot;P)}{\partial P}
        */
        PETScWrappers::MPI::SparseMatrix jacobian_matrix_dFdP;
        /*
            \frac{\partial r(u,P)}{\partial u}
        */
        PETScWrappers::MPI::SparseMatrix jacobian_matrix_dRdU;
        /*
            \frac{\partial r(u,P)}{\partial P}
        */
        PETScWrappers::MPI::SparseMatrix jacobian_matrix_dRdP;
        /*
            \lambda
        */
        PETScWrappers::MPI::Vector adjoint_lambda;
        /*
            \mu
        */
        PETScWrappers::MPI::Vector adjoint_mu;
    // Helmholtz equation.
        IndexSet locally_owned_dofs_filter;
        IndexSet locally_relevant_dofs_filter;

        AffineConstraints<double> helmholtz_constraints;
        /*
            Helmholtz matrix
        */
        PETScWrappers::MPI::SparseMatrix helmholtz_matrix;
        /*
            Helmholtz rhs
        */
        PETScWrappers::MPI::Vector helmholtz_rhs;
        /*
            Helmholtz solution
        */
       PETScWrappers::MPI::Vector helmholtz_solution;
       PETScWrappers::MPI::Vector helmholtz_locally_solution;
    // Output.
    ConditionalOStream pcout;
    TimerOutput computing_timer;
    // Design variables.
        PETScWrappers::MPI::Vector simp_rho;
        PETScWrappers::MPI::Vector simp_rho_filter;

        PETScWrappers::MPI::Vector simp_rho_max;
        PETScWrappers::MPI::Vector simp_rho_min;

        PETScWrappers::MPI::Vector cell_volume;

        PETScWrappers::MPI::Vector              Object_Function_Derivative;
        std::vector<PETScWrappers::MPI::Vector> Constraint_Function_Derivative;
    // MMA.
    Deal_II_MMA_Solver MMA_Solver;
    // Filter radius.
    double        Rmin;
    // Fixed parameters.
    double        epsimin = 1.e-9;
    double        volfrac = 0.3;
    // Material parameters: only high-thermal-conductivity material and
    // phase-change material are considered.
    // 0 - phase-change material, 1 - high-thermal-conductivity material.
    std::vector<double> heat_conductivity={10,100};
    std::vector<double> density={100,100};
    std::vector<double> heat_capacity={100,100};
    // SIMP interpolation.
    SIMP MySimp;
    // Phase-change function.
    PCM  MyPcm;
};