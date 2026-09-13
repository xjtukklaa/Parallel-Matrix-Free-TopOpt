#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /* TopOptHeatFlow 类初始化 */

    TopOptHeatFlow::TopOptHeatFlow(const unsigned int degree,
                                   const unsigned int top_degree)
        : start_point(PRM::start_point),
          obj_guess(PRM::obj_guess),
          flow_cstr(PRM::flow_cstr),
          vol_cstr(PRM::vol_cstr),
          field_cstr(PRM::field_cstr),
          loop_counter(0),

          field_if_epsilon(PRM::field_if_epsilon),
          field_plus_epsilon(PRM::field_plus_epsilon),

          obj_value(0.0),
          cstr_flow_value(0.0),
          cstr_vol_value(0.0),
          cstr_field_value(0.0),

          L_refer(PRM::L_refer),
          U_refer(PRM::U_refer),
          rho_refer(PRM::rho_refer),
          viscosity_refer(PRM::viscosity_refer),
          Re_refer(PRM::Re_refer),
          Da_refer(PRM::Da_refer),

          t_in_refer(PRM::t_in_refer),
          t_Q_refer(PRM::t_Q_refer),
          capacity_refer(PRM::capacity_refer),
          k_f_refer(PRM::k_f_refer),
          k_s_refer(PRM::k_s_refer),
          Q_refer(PRM::Q_refer),
          heatflux_refer(PRM::heatflux_refer),
          Pr_refer(PRM::Pr_refer),

          rho(PRM::rho),
          viscosity(PRM::viscosity),
          permeability(0.0),
          capacity(PRM::capacity),
          thermal_conductivity(0.0),
          heatsource_coeff(0.0),
          convection_coeff(0.0),
          t_Q(PRM::t_Q),
          Q0(PRM::Q0),
          heatflux0(PRM::heatflux0),
          uniform_element_size(0.0),
          filter_radius(PRM::filter_radius),
          heaviside_beta(PRM::heaviside_beta),
          heaviside_yita(PRM::heaviside_yita),
          heaviside_beta_equivalent(PRM::heaviside_beta_equivalent),
          heaviside_yita_equivalent(PRM::heaviside_yita_equivalent),

          relax_heaviside_beta(PRM::relax_heaviside_beta_step_init,
                               PRM::relax_heaviside_beta_step_size,
                               PRM::relax_heaviside_beta_para_start,
                               PRM::relax_heaviside_beta_para_end,
                               PRM::relax_heaviside_beta_para_coeff),
          relax_heaviside_beta_equivalent(PRM::relax_heaviside_beta_equivalent_step_init,
                                          PRM::relax_heaviside_beta_equivalent_step_size,
                                          PRM::relax_heaviside_beta_equivalent_para_start,
                                          PRM::relax_heaviside_beta_equivalent_para_end,
                                          PRM::relax_heaviside_beta_equivalent_para_coeff),
          relax_da_refer(PRM::relax_da_refer_step_init,
                         PRM::relax_da_refer_step_size,
                         PRM::relax_da_refer_para_start,
                         PRM::relax_da_refer_para_end,
                         PRM::relax_da_refer_para_coeff),

          permeability_fluid(PRM::permeability_fluid),
          permeability_solid(PRM::permeability_solid),
          permeability_penal(PRM::permeability_penal),
          thermal_cond_fluid(PRM::thermal_cond_fluid),
          thermal_cond_solid(PRM::thermal_cond_solid),
          thermal_cond_penal(PRM::thermal_cond_penal),
          heatsource_coeff_solid(PRM::heatsource_coeff_solid),
          heatsource_coeff_penal(PRM::heatsource_coeff_penal),
          convection_coeff_fluid(PRM::convection_coeff_fluid),
          convection_coeff_penal(PRM::convection_coeff_penal),
          material_interpolate(permeability_fluid, permeability_solid,
                               permeability_penal, thermal_cond_fluid,
                               thermal_cond_solid, thermal_cond_penal,
                               heatsource_coeff_solid, heatsource_coeff_penal,
                               convection_coeff_fluid, convection_coeff_penal),
          heaviside(heaviside_beta, heaviside_yita),
          heaviside_equivalent(heaviside_beta_equivalent, heaviside_yita_equivalent),

          domain_volume(3, 0.0),
          boundary_area(5, 0.0),

          degree(degree),
          top_degree(top_degree),
          top_equivalent_degree(0),
          quadrature_degree(degree),
          mpi_communicator(MPI_COMM_WORLD),
          triangulation(mpi_communicator,
                        typename Triangulation<dim>::MeshSmoothing(
                            Triangulation<dim>::smoothing_on_refinement |
                            Triangulation<dim>::smoothing_on_coarsening)),

          filter_mapping(MappingFE<dim>(FE_Q<dim>(std::max(degree, top_degree)))),
          top_mapping(),

          top_cg_mapping(MappingQ1<dim>()),
          top_dg_mapping(MappingQ1<dim>()),
          top_equivalent_mapping(MappingQ1<dim>()),

          ns_fe(FE_Q<dim>(degree), dim, FE_Q<dim>(degree), 1),
          temp_fe(FE_Q<dim>(degree), 1),
          ad_temp_fe(FE_Q<dim>(degree), 1),
          ad_ns_fe(FE_Q<dim>(degree), dim, FE_Q<dim>(degree), 1),

          top_cg_fe(FE_Q<dim>(top_degree == 0 ? 1 : top_degree), FE_DGQ<dim>(0)),
          top_dg_fe(FE_DGQ<dim>(top_degree), FE_DGQ<dim>(0)),
          top_equivalent_fe(FE_DGQ<dim>(top_equivalent_degree), FE_DGQ<dim>(0)),
          filter_fe(FE_Q<dim>(std::max(degree, top_degree)), FE_Nothing<dim>()),

          ns_dof_handler(triangulation),
          temp_dof_handler(triangulation),
          ad_temp_dof_handler(triangulation),
          ad_ns_dof_handler(triangulation),
          top_dof_handler(triangulation),
          top_equivalent_dof_handler(triangulation),
          filter_dof_handler(triangulation),

          mma(nullptr),
          obj_cstr_init_step_value(4, 0.0),

          iteration_method(PRM::iteration_method),
          objective_function(PRM::objective_function),
          flow_constraint_function(PRM::flow_constraint_function),
          field_constraint_function(PRM::field_constraint_function),
          lagrange_function_type(PRM::lagrange_obj_function),

          pcout(std::cout,
                Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
          computing_timer(mpi_communicator,
                          pcout,
                          TimerOutput::never,
                          TimerOutput::wall_times)
    {
        pcout << "******************************************************************************" << std::endl;
        pcout << "The dimensionless parameter is ...." << std::endl
              << "Re = " << Re_refer << std::endl
              << "Da = " << Da_refer << std::endl
              << "Pr = " << Pr_refer << std::endl;
    }

    /* 析构函数 */
    TopOptHeatFlow::~TopOptHeatFlow()
    {

        VecDestroy(&dfdx_petsc);
        VecDestroyVecs(n_constraints, &dgdx_petsc);
        delete mma;
    }
}