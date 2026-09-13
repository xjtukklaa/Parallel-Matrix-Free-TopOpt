#ifndef TOP_OPT_HEAT_FLOW
#define TOP_OPT_HEAT_FLOW

#include "../include/dealiipackage.h"
#include "../include/mma.h"
#include "../include/prm.h"
#include "../include/material_interpolate.h"
#include "../include/heaviside.h"
#include "../include/parameter_relax.h"
#include "../include/stabilization_parameter.h"
#include "../include/boundary.h"
#include "../include/change_vector_types.h"

namespace TopOpt
{
    using namespace dealii;

    constexpr int dim = 2;

    class TopOptHeatFlow
    {
    public:
        TopOptHeatFlow(const unsigned int degree,
                       const unsigned int top_degree);
        void run(const unsigned int loop, const unsigned int refinement);
        ~TopOptHeatFlow();

    private:
        void make_grid();
        void define_grid();
        void get_grid_information();

        void setup_dofs();
        void initialize_system();

        void reinit_mma(const bool is_initial_step);

        void update_current_parameters();

        void reset_topology(const bool is_initial_step);
        void filter_assemble(const bool assemble_matrix,
                             const LA::MPI::Vector &top_vector_relevant);
        void filter_solve();
        void filter_transfer_to_top(const bool is_sens,
                                    LA::MPI::Vector &top_vector_distributed,
                                    LA::MPI::Vector &top_vector_relevant);
        void xphys_heaviside();
        void xphys_heaviside_conservation();
        void xphys_equivalent();
        void xphys_range_constraint(LA::MPI::Vector &top_vector_distributed,
                                    LA::MPI::Vector &top_vector_relevant);

        void ns_assemble(const bool assemble_matrix);
        void ns_assemble_system();
        void ns_assemble_rhs();
        void ns_solve();

        void ns_nls_assemble(const bool is_jacobian, const LA::MPI::Vector &evaluation_point, LA::MPI::Vector &residual);
        void ns_nls_assemble_with_residual_linearization(const bool is_jacobian, const LA::MPI::Vector &evaluation_point, LA::MPI::Vector &residual);
        void ns_nls_compute_residual(const LA::MPI::Vector &evaluation_point, LA::MPI::Vector &residual);
        void ns_nls_compute_jacobian(const LA::MPI::Vector &evaluation_point);
        void ns_nls_compute_jacobian_and_initialize_preconditioner(const LA::MPI::Vector &evaluation_point);
        void ns_nls_solve(const LA::MPI::Vector &rhs, LA::MPI::Vector &solution);

        void newton_iteration(const double tolerance,
                              const unsigned int max_n_line_searches,
                              const unsigned int max_n_refinements);
        void compute_initial_guess(double Re_init, double step_size);

        void ns_nonlinear_solver(const bool is_init_step);

        void temp_assemble();
        void temp_solve();

        void ad_continuous_temp_assemble();
        void ad_continuous_temp_assemble_with_residual_linearization();
        void ad_continuous_temp_solve();

        void ad_continuous_ns_assemble(const PRM::LagrangeFunctionType lag_function,
                                       const bool assemble_matrix);
        void ad_continuous_ns_assemble_with_residual_linearization(
            const PRM::LagrangeFunctionType lag_function,
            const bool assemble_matrix);
        void ad_continuous_ns_solve(const PRM::LagrangeFunctionType lag_function);

        void sens_assemble();
        void sensitivity_dheaviside();
        void finite_difference_sensitivity_check(
            const double x1,
            const double x2,
            const double y1,
            const double y2,
            const double perturbation,
            const unsigned int max_cells);
        void output_finite_difference_sensitivity(
            const LA::MPI::Vector &selected_distributed,
            const LA::MPI::Vector &fd_obj_distributed,
            const LA::MPI::Vector &fd_flow_distributed,
            const LA::MPI::Vector &fd_vol_distributed,
            const unsigned int loop_num,
            const double perturbation);

        void output_results(const unsigned int refinement_cycle,
                            const unsigned int loop_num);
        void update_function_values(const bool print_values = true);
        void topology_update(double &change);

        void estimate_topology_boundary(Vector<float> &estimated_error_per_cell);
        void refine_mesh();

        void topology_iteration(const double top_min_change,
                                const unsigned int top_max_n_loops,
                                const unsigned int top_max_n_refinements,
                                unsigned int output_interval);

        double start_point;
        double obj_guess;
        double flow_cstr;
        double vol_cstr;
        double field_cstr;
        unsigned int loop_counter;

        double field_if_epsilon;
        double field_plus_epsilon;

        double obj_value;
        double cstr_flow_value;
        double cstr_vol_value;
        double cstr_field_value;

        double L_refer;
        double U_refer;
        double rho_refer;
        double viscosity_refer;
        double Re_refer;
        double Da_refer;

        double t_in_refer;
        double t_Q_refer;
        double capacity_refer;
        double k_f_refer;
        double k_s_refer;
        double Q_refer;
        double heatflux_refer;
        double Pr_refer;

        double rho;
        double viscosity;
        double permeability;
        double capacity;
        double thermal_conductivity;
        double heatsource_coeff;
        double convection_coeff;
        double t_Q;
        double Q0;
        double heatflux0;
        double uniform_element_size;
        double filter_radius;
        double heaviside_beta;
        double heaviside_yita;
        double heaviside_beta_equivalent;
        double heaviside_yita_equivalent;

        ParameterRelax relax_heaviside_beta;
        ParameterRelax relax_heaviside_beta_equivalent;
        ParameterRelax relax_da_refer;

        double permeability_fluid;
        double permeability_solid;
        double permeability_penal;
        double thermal_cond_fluid;
        double thermal_cond_solid;
        double thermal_cond_penal;
        double heatsource_coeff_solid;
        double heatsource_coeff_penal;
        double convection_coeff_fluid;
        double convection_coeff_penal;
        MaterialInterpolate material_interpolate;
        Heaviside heaviside;
        Heaviside heaviside_equivalent;

        std::vector<double> domain_volume;
        std::vector<double> boundary_area;

        const unsigned int degree;
        const unsigned int top_degree;
        const unsigned int top_equivalent_degree;
        const unsigned int quadrature_degree;
        MPI_Comm mpi_communicator;

        parallel::distributed::Triangulation<dim> triangulation;
        hp::MappingCollection<dim> filter_mapping;
        hp::MappingCollection<dim> top_mapping;
        hp::MappingCollection<dim> top_cg_mapping;
        hp::MappingCollection<dim> top_dg_mapping;
        hp::MappingCollection<dim> top_equivalent_mapping;
        FESystem<dim> ns_fe;
        FESystem<dim> temp_fe;
        FESystem<dim> ad_temp_fe;
        FESystem<dim> ad_ns_fe;

        hp::FECollection<dim> top_cg_fe;
        hp::FECollection<dim> top_dg_fe;
        hp::FECollection<dim> top_equivalent_fe;
        hp::FECollection<dim> filter_fe;

        DoFHandler<dim> ns_dof_handler;
        DoFHandler<dim> temp_dof_handler;
        DoFHandler<dim> ad_temp_dof_handler;
        DoFHandler<dim> ad_ns_dof_handler;
        DoFHandler<dim> top_dof_handler;
        DoFHandler<dim> top_equivalent_dof_handler;
        DoFHandler<dim> filter_dof_handler;

        IndexSet ns_owned_partitioning;
        IndexSet ns_relevant_partitioning;
        IndexSet temp_owned_partitioning;
        IndexSet temp_relevant_partitioning;
        IndexSet ad_temp_owned_partitioning;
        IndexSet ad_temp_relevant_partitioning;
        IndexSet ad_ns_owned_partitioning;
        IndexSet ad_ns_relevant_partitioning;
        IndexSet top_owned_partitioning;
        IndexSet top_relevant_partitioning;
        IndexSet top_equivalent_owned_partitioning;
        IndexSet top_equivalent_relevant_partitioning;
        IndexSet filter_owned_partitioning;
        IndexSet filter_relevant_partitioning;

        AffineConstraints<double> ns_zero_constraints;
        AffineConstraints<double> ns_nonzero_constraints;
        AffineConstraints<double> temp_constraints;
        AffineConstraints<double> ad_temp_constraints;
        AffineConstraints<double> ad_ns_constraints;
        AffineConstraints<double> top_constraints;
        AffineConstraints<double> top_equivalent_constraints;
        AffineConstraints<double> filter_constraints;

        LA::MPI::SparseMatrix ns_system_matrix;
        LA::MPI::SparseMatrix temp_system_matrix;
        LA::MPI::SparseMatrix ad_temp_system_matrix;
        LA::MPI::SparseMatrix ad_ns_system_matrix;
        LA::MPI::SparseMatrix filter_system_matrix;

        LA::MPI::Vector ns_solution_distributed;
        LA::MPI::Vector ns_solution_relevant;
        LA::MPI::Vector ns_newton_update_distributed;
        LA::MPI::Vector ns_newton_update_relevant;
        LA::MPI::Vector ns_evaluation_distributed;
        LA::MPI::Vector ns_evaluation_relevant;
        LA::MPI::Vector ns_system_rhs;
        LA::MPI::Vector temp_solution_distributed;
        LA::MPI::Vector temp_solution_relevant;
        LA::MPI::Vector temp_system_rhs;

        LA::MPI::Vector ad_temp_solution_distributed;
        LA::MPI::Vector ad_temp_solution_relevant;
        LA::MPI::Vector ad_temp_system_rhs;
        LA::MPI::Vector ad_ns_solution_distributed;
        LA::MPI::Vector ad_ns_solution_relevant;
        LA::MPI::Vector ad_ns_flow_solution_distributed;
        LA::MPI::Vector ad_ns_flow_solution_relevant;
        LA::MPI::Vector ad_ns_field_solution_distributed;
        LA::MPI::Vector ad_ns_field_solution_relevant;
        LA::MPI::Vector ad_ns_system_rhs;

        LA::MPI::Vector xphys_values_distributed;
        LA::MPI::Vector xphys_values_relevant;
        LA::MPI::Vector xphys_filter_distributed;
        LA::MPI::Vector xphys_filter_relevant;
        LA::MPI::Vector xphys_heaviside_distributed;
        LA::MPI::Vector xphys_heaviside_relevant;
        LA::MPI::Vector xphys_equivalent_distributed;
        LA::MPI::Vector xphys_equivalent_relevant;

        LA::MPI::Vector sens_obj_distributed;
        LA::MPI::Vector sens_obj_relevant;
        LA::MPI::Vector sens_flow_distributed;
        LA::MPI::Vector sens_flow_relevant;
        LA::MPI::Vector sens_vol_distributed;
        LA::MPI::Vector sens_vol_relevant;
        LA::MPI::Vector sens_field_distributed;
        LA::MPI::Vector sens_field_relevant;

        LA::MPI::Vector sens_obj_dheaviside_distributed;
        LA::MPI::Vector sens_obj_dheaviside_relevant;
        LA::MPI::Vector sens_flow_dheaviside_distributed;
        LA::MPI::Vector sens_flow_dheaviside_relevant;
        LA::MPI::Vector sens_vol_dheaviside_distributed;
        LA::MPI::Vector sens_vol_dheaviside_relevant;
        LA::MPI::Vector sens_field_dheaviside_distributed;
        LA::MPI::Vector sens_field_dheaviside_relevant;

        LA::MPI::Vector sens_obj_dfilter_distributed;
        LA::MPI::Vector sens_obj_dfilter_relevant;
        LA::MPI::Vector sens_flow_dfilter_distributed;
        LA::MPI::Vector sens_flow_dfilter_relevant;
        LA::MPI::Vector sens_vol_dfilter_distributed;
        LA::MPI::Vector sens_vol_dfilter_relevant;
        LA::MPI::Vector sens_field_dfilter_distributed;
        LA::MPI::Vector sens_field_dfilter_relevant;

        LA::MPI::Vector filter_solution_distributed;
        LA::MPI::Vector filter_solution_relevant;
        LA::MPI::Vector filter_system_rhs;

        LA::MPI::Vector xold1_distributed, xold1_relevant;
        LA::MPI::Vector xold2_distributed, xold2_relevant;
        LA::MPI::Vector U_distributed, U_relevant;
        LA::MPI::Vector L_distributed, L_relevant;

        Vec xval_petsc;
        Vec xold1_petsc, xold2_petsc;
        Vec xmin_petsc, xmax_petsc;
        Vec dfdx_petsc;
        Vec *dgdx_petsc;
        PetscInt n_variables_global;
        PetscInt n_constraints;
        MMA *mma;
        std::vector<double> obj_cstr_init_step_value;

        PRM::IterationMethod iteration_method;
        PRM::ObjectiveFunction objective_function;
        PRM::FlowConstraintFunction flow_constraint_function;
        PRM::FieldConstraintFunction field_constraint_function;
        PRM::LagrangeFunctionType lagrange_function_type;

        ConditionalOStream pcout;
        TimerOutput computing_timer;
    };

}

#endif