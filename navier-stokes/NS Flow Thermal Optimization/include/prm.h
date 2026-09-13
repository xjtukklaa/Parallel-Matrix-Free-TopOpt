#ifndef PRM_H
#define PRM_H

#include <deal.II/base/parameter_handler.h>
#include <iostream>

namespace PRM
{
  using namespace dealii;

  enum IterationMethod
  {
    newton_method,
    oseen_method,
    stokes_method
  };
  enum ObjectiveFunction
  {
    mean_temp_objective,
    abs_u_gradt_objective,
    norm_gradt_objective,
    mean_solid_temp_objective,
    thermal_compliance_objective,
    heat_exchange_objective,
    mean_boundary_temp_objective,
    xphys_abs_cos_u_grad_temp_objective
  };
  enum FlowConstraintFunction
  {
    flow_dissipation_constraint,
    inlet_pressure_constraint,
    pump_power_constraint,
    only_volume_constraint
  };

  enum FieldConstraintFunction
  {
    no_field_constraints,
    norm_xphys_heaviside_gradients_constraint,
    norm_u_grad_xphys_constraint,
    abs_cos_u_grad_xphys_constraint,
    xphys_abs_cos_u_grad_xphys_constraint,
    fnorm_xphys_filter_hessians_constraint,
    norm_xphys_filter_gradients_constraint
  };

  enum ManufactureConstraintFunction
  {
    minimum_length_constraint,
    maximum_length_constraint,
    overhang_angle_constraint
  };

  enum LagrangeFunctionType
  {
    lagrange_obj_function,
    lagrange_flow_cstr_function,
    lagrange_vol_cstr_function,
    lagrange_field_cstr_function,
    lagrange_length_cstr_function
  };

  enum DimensionlessMethod
  {
    dimensionless_on,
    dimensionless_off
  };

  enum NormalizationMethod
  {
    normalization_on,
    normalization_off
  };

  struct ParameterSet
  {
    IterationMethod iteration_method = newton_method;
    ObjectiveFunction objective_function = heat_exchange_objective;
    FlowConstraintFunction flow_constraint_function = flow_dissipation_constraint;
    FieldConstraintFunction field_constraint_function = no_field_constraints;
    DimensionlessMethod dimensionless_method = dimensionless_on;
    NormalizationMethod normalization_method = normalization_on;

    double start_point = 0.75;
    double obj_guess = 1340.0;
    double vol_cstr = 0.75;
    double flow_cstr = 5.0;
    double field_cstr = 0.25;
    double field_if_epsilon = 1e-8;
    double field_plus_epsilon = 0.0;

    unsigned int degree = 1;
    unsigned int top_degree = 2;
    unsigned int loop = 120;
    unsigned int refinement = 0;

    unsigned int relax_heaviside_beta_step_init = 20;
    unsigned int relax_heaviside_beta_step_size = 20;
    double relax_heaviside_beta_para_start = 1.0;
    double relax_heaviside_beta_para_end = 8.0;
    double relax_heaviside_beta_para_coeff = 2.0;

    unsigned int relax_heaviside_beta_equivalent_step_init = 80;
    unsigned int relax_heaviside_beta_equivalent_step_size = 20;
    double relax_heaviside_beta_equivalent_para_start = 1.0;
    double relax_heaviside_beta_equivalent_para_end = 1.0;
    double relax_heaviside_beta_equivalent_para_coeff = 2.0;

    unsigned int relax_da_refer_step_init = 20;
    unsigned int relax_da_refer_step_size = 20;
    double relax_da_refer_para_start = 1e-3;
    double relax_da_refer_para_end = 1e-4;
    double relax_da_refer_para_coeff = 0.4;

    double filter_radius_dml = 0.4;
    double filter_radius = 0.4;
    double heaviside_beta = 1.0;
    double heaviside_yita = 0.5;
    double heaviside_beta_equivalent = 1.0;
    double heaviside_yita_equivalent = 0.5;

    double L_refer = 2e-3;
    double U_refer = 0.1;
    double rho_refer = 1000.0;
    double viscosity_refer = 1e-3;
    double Re_refer = rho_refer * U_refer * L_refer / viscosity_refer;
    double Da_refer = 1e-3;

    double t_in_refer = 10.0;
    double t_Q_refer = 30.0;
    double capacity_refer = 4183.0;
    double k_f_refer = 0.6;
    double k_s_refer = 180.0;
    double Q_refer = 0.0;
    double heatflux_refer = 0.0;
    double heatsource_coeff_solid_dml = 1e4;
    double Pr_refer = viscosity_refer * capacity_refer / k_f_refer;

    double rho = 1.0;
    double viscosity = 1.0 / Re_refer;
    double capacity = Re_refer * Pr_refer;
    double t_Q = 1.0;
    double Q0 = L_refer * L_refer * Q_refer / (k_f_refer * (t_Q_refer - t_in_refer));
    double heatflux0 = L_refer * heatflux_refer / (k_f_refer * (t_Q_refer - t_in_refer));
    double inlet_velocity = 1.0;
    double inlet_temperature = 0.0;

    unsigned int pre_refinement = 1;

    double permeability_fluid = 0.0;
    double permeability_solid = (1.0 + (1.0 / Re_refer)) / Da_refer;
    double permeability_penal = 0.1;
    double thermal_cond_fluid = 1.0;
    double thermal_cond_solid = k_s_refer / k_f_refer;
    double thermal_cond_penal = 1.0;
    double heatsource_coeff_solid = 1e3;
    double heatsource_coeff_penal = 1.0;
    double convection_coeff_fluid = 1.0;
    double convection_coeff_penal = 1.0;
  };

  ParameterSet make_default_parameters();

  /**********************************************************************************************************/


  /**********************************************************************************************************/
  /* Topology optimization method parameters */

  extern IterationMethod iteration_method;
  extern ObjectiveFunction objective_function;
  extern FlowConstraintFunction flow_constraint_function;
  extern FieldConstraintFunction field_constraint_function;

  extern DimensionlessMethod dimensionless_method;
  extern NormalizationMethod normalization_method;

  /**********************************************************************************************************/
  /* 1. Topology optimization start parameters */

  extern double start_point;
  extern double obj_guess;
  extern double vol_cstr;
  extern double flow_cstr;
  extern double field_cstr;

  extern double field_if_epsilon;
  extern double field_plus_epsilon;






  extern unsigned int degree;
  extern unsigned int top_degree;
  extern unsigned int loop;
  extern unsigned int refinement;

  /**********************************************************************************************************/
  /* 2. Topology optimization relax parameters */
  extern unsigned int relax_heaviside_beta_step_init;
  extern unsigned int relax_heaviside_beta_step_size;
  extern double relax_heaviside_beta_para_start;
  extern double relax_heaviside_beta_para_end;
  extern double relax_heaviside_beta_para_coeff;

  extern unsigned int relax_heaviside_beta_equivalent_step_init;
  extern unsigned int relax_heaviside_beta_equivalent_step_size;
  extern double relax_heaviside_beta_equivalent_para_start;
  extern double relax_heaviside_beta_equivalent_para_end;
  extern double relax_heaviside_beta_equivalent_para_coeff;

  extern unsigned int relax_da_refer_step_init;
  extern unsigned int relax_da_refer_step_size;
  extern double relax_da_refer_para_start;
  extern double relax_da_refer_para_end;
  extern double relax_da_refer_para_coeff;

  /**********************************************************************************************************/
  /* Topology optimization filter and projection parameters */

  extern double filter_radius_dml;

  extern double filter_radius;
  extern double heaviside_beta;
  extern double heaviside_yita;
  extern double heaviside_beta_equivalent;
  extern double heaviside_yita_equivalent;

  /**********************************************************************************************************/
  /* Topology optimization boundary parameters */

  extern double L_refer;
  extern double U_refer;
  extern double rho_refer;
  extern double viscosity_refer;
  extern double Re_refer;
  extern double Da_refer;

  extern double t_in_refer;
  extern double t_Q_refer;
  extern double capacity_refer;
  extern double k_f_refer;
  extern double k_s_refer;
  extern double Q_refer;
  extern double heatflux_refer;
  extern double heatsource_coeff_solid_dml;
  extern double Pr_refer;

  extern double rho;
  extern double viscosity;
  extern double capacity;
  extern double t_Q;
  extern double Q0;
  extern double heatflux0;
  extern double inlet_velocity;
  extern double inlet_temperature;






  /**********************************************************************************************************/
  /* Topology optimization mesh parameters */

  extern unsigned int pre_refinement;


  /**********************************************************************************************************/
  /* Topology optimization material penalty parameters */

  extern double permeability_fluid;
  extern double permeability_solid;
  extern double permeability_penal;
  extern double thermal_cond_fluid;
  extern double thermal_cond_solid;
  extern double thermal_cond_penal;
  extern double heatsource_coeff_solid;
  extern double heatsource_coeff_penal;
  extern double convection_coeff_fluid;
  extern double convection_coeff_penal;

  /**********************************************************************************************************/
  /* Definition of a parameter modifier */
  class ParameterModifier : public Subscriptor
  {
  public:
    ParameterModifier(ParameterHandler &);
    void read_parameters(const std::string &);
    void write_parameters();

  private:
    void declare_parameters();
    ParameterHandler &prm;
  };
}

#endif