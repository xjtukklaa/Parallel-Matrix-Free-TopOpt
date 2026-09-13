#include "../include/prm.h"

namespace PRM
{
    ParameterSet make_default_parameters()
    {
        ParameterSet parameters;

        parameters.iteration_method = newton_method;
        parameters.objective_function = heat_exchange_objective;
        parameters.flow_constraint_function = flow_dissipation_constraint;
        parameters.field_constraint_function = no_field_constraints;
        parameters.dimensionless_method = dimensionless_on;
        parameters.normalization_method = normalization_on;

        parameters.start_point = 0.75;
        parameters.obj_guess = 1340.0;
        parameters.vol_cstr = 0.75;
        parameters.flow_cstr = 5.0;
        parameters.field_cstr = 0.25;
        parameters.field_if_epsilon = 1e-8;
        parameters.field_plus_epsilon = 0.0;

        parameters.degree = 1;
        parameters.top_degree = 2;
        parameters.loop = 120;
        parameters.refinement = 0;

        parameters.relax_heaviside_beta_step_init = 20;
        parameters.relax_heaviside_beta_step_size = 20;
        parameters.relax_heaviside_beta_para_start = 1.0;
        parameters.relax_heaviside_beta_para_end = 8.0;
        parameters.relax_heaviside_beta_para_coeff = 2.0;

        parameters.relax_heaviside_beta_equivalent_step_init = 80;
        parameters.relax_heaviside_beta_equivalent_step_size = 20;
        parameters.relax_heaviside_beta_equivalent_para_start = 1.0;
        parameters.relax_heaviside_beta_equivalent_para_end = 1.0;
        parameters.relax_heaviside_beta_equivalent_para_coeff = 2.0;

        parameters.relax_da_refer_step_init = 20;
        parameters.relax_da_refer_step_size = 20;
        parameters.relax_da_refer_para_start = 1e-3;
        parameters.relax_da_refer_para_end = 1e-4;
        parameters.relax_da_refer_para_coeff = 0.4;

        parameters.filter_radius_dml = 0.4;
        parameters.filter_radius = 0.4;
        parameters.heaviside_beta = parameters.relax_heaviside_beta_para_start;
        parameters.heaviside_yita = 0.5;
        parameters.heaviside_beta_equivalent = parameters.relax_heaviside_beta_equivalent_para_start;
        parameters.heaviside_yita_equivalent = 0.5;

        parameters.L_refer = 2e-3;
        parameters.U_refer = 0.1;
        parameters.rho_refer = 1000.0;
        parameters.viscosity_refer = 1e-3;
        parameters.Re_refer = parameters.rho_refer * parameters.U_refer * parameters.L_refer / parameters.viscosity_refer;
        parameters.Da_refer = parameters.relax_da_refer_para_start;
        parameters.t_in_refer = 10.0;
        parameters.t_Q_refer = 30.0;
        parameters.capacity_refer = 4183.0;
        parameters.k_f_refer = 0.6;
        parameters.k_s_refer = 180.0;
        parameters.Q_refer = 0.0;
        parameters.heatflux_refer = 0.0;
        parameters.heatsource_coeff_solid_dml = 1e4;
        parameters.Pr_refer = parameters.viscosity_refer * parameters.capacity_refer / parameters.k_f_refer;

        parameters.rho = 1.0;
        parameters.viscosity = 1.0 / parameters.Re_refer;
        parameters.capacity = parameters.Re_refer * parameters.Pr_refer;
        parameters.t_Q = 1.0;
        parameters.Q0 = parameters.L_refer * parameters.L_refer * parameters.Q_refer /
                       (parameters.k_f_refer * (parameters.t_Q_refer - parameters.t_in_refer));
        parameters.heatflux0 = parameters.L_refer * parameters.heatflux_refer /
                              (parameters.k_f_refer * (parameters.t_Q_refer - parameters.t_in_refer));
        parameters.inlet_velocity = 1.0;
        parameters.inlet_temperature = 0.0;

        parameters.pre_refinement = 1;

        parameters.permeability_fluid = 0.0;
        parameters.permeability_solid = (1.0 + (1.0 / parameters.Re_refer)) / parameters.Da_refer;
        parameters.permeability_penal = 0.1;
        parameters.thermal_cond_fluid = 1.0;
        parameters.thermal_cond_solid = parameters.k_s_refer / parameters.k_f_refer;
        parameters.thermal_cond_penal = 1.0;
        parameters.heatsource_coeff_solid = 1e3;
        parameters.heatsource_coeff_penal = 1.0;
        parameters.convection_coeff_fluid = 1.0;
        parameters.convection_coeff_penal = 1.0;

        return parameters;
    }

    /**********************************************************************************************************/


    /**********************************************************************************************************/
    /* Topology optimization method parameters */

    IterationMethod iteration_method = newton_method;
    ObjectiveFunction objective_function = heat_exchange_objective;
    FlowConstraintFunction flow_constraint_function = flow_dissipation_constraint;
    FieldConstraintFunction field_constraint_function = no_field_constraints;

    DimensionlessMethod dimensionless_method = dimensionless_on;
    NormalizationMethod normalization_method = normalization_on;

    /**********************************************************************************************************/
    /* 1. Topology optimization start parameters */

    double start_point = 0.75;
    double obj_guess = 1340;
    double vol_cstr = 0.75;
    double flow_cstr = 5.0;
    double field_cstr = 0.25;

    double field_if_epsilon = 1e-8;
    double field_plus_epsilon = 0.0;






    unsigned int degree = 1;
    unsigned int top_degree = 2;
    unsigned int loop = 120;
    unsigned int refinement = 0;

    /**********************************************************************************************************/
    /* 2. Topology optimization relax parameters */
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

    /**********************************************************************************************************/
    /* Topology optimization filter and projection parameters */

    double filter_radius_dml = 0.4;

    double filter_radius = 0.4;
    double heaviside_beta = relax_heaviside_beta_para_start;
    double heaviside_yita = 0.5;
    double heaviside_beta_equivalent = relax_heaviside_beta_equivalent_para_start;
    double heaviside_yita_equivalent = 0.5;

    /**********************************************************************************************************/
    /* Topology optimization boundary parameters */

    double L_refer = 2e-3;
    double U_refer = 0.1;
    double rho_refer = 1000.0;
    double viscosity_refer = 1e-3;
    double Re_refer = rho_refer * U_refer * L_refer / viscosity_refer;
    double Da_refer = relax_da_refer_para_start;

    double t_in_refer = 10.0;
    double t_Q_refer = 30.0;
    double capacity_refer = 4183;
    double k_f_refer = 0.6;
    double k_s_refer = 180;
    double Q_refer = 0.0;
    double heatflux_refer = 0.0;
    double heatsource_coeff_solid_dml = 1e3;
    double Pr_refer = viscosity_refer * capacity_refer / k_f_refer;

    double rho = 1.0;
    double viscosity = 1.0 / Re_refer;
    double capacity = Re_refer * Pr_refer;
    double t_Q = 1.0;
    double Q0 = L_refer * L_refer * Q_refer / (k_f_refer * (t_Q_refer - t_in_refer));
    double heatflux0 = L_refer * heatflux_refer / (k_f_refer * (t_Q_refer - t_in_refer));
    double inlet_velocity = 1.0;
    double inlet_temperature = 0.0;






    /**********************************************************************************************************/
    /* Topology optimization mesh parameters */

    unsigned int pre_refinement = 1;


    /**********************************************************************************************************/
    /* Topology optimization material penalty parameters */

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

    ParameterModifier::ParameterModifier(ParameterHandler &parameterhandler)
        : prm(parameterhandler)
    {
    }

    void PRM::ParameterModifier::declare_parameters()
    {
        prm.enter_subsection("Topology optimization method parameters");
        {
            prm.declare_entry("iteration_method",
                              "newton_method",
                              Patterns::Selection("newton_method|oseen_method|stokes_method"),
                              "The iteration method for solving Navier-Stokes equations (newton_method|oseen_method|stokes_method)");
            prm.declare_entry("objective_function",
                              "heat_exchange_objective",
                              Patterns::Selection("mean_temp_objective|abs_u_gradt_objective|norm_gradt_objective|mean_solid_temp_objective|thermal_compliance_objective|heat_exchange_objective|mean_boundary_temp_objective|xphys_abs_cos_u_grad_temp_objective"),
                              "The objective function (mean_temp_objective|abs_u_gradt_objective|norm_gradt_objective|mean_solid_temp_objective|thermal_compliance_objective|heat_exchange_objective|mean_boundary_temp_objective|xphys_abs_cos_u_grad_temp_objective)");
            prm.declare_entry("flow_constraint_function",
                              "flow_dissipation_constraint",
                              Patterns::Selection("flow_dissipation_constraint|inlet_pressure_constraint|pump_power_constraint|only_volume_constraint"),
                              "The flow constraint function (flow_dissipation_constraint|inlet_pressure_constraint|pump_power_constraint|only_volume_constraint)");
            prm.declare_entry("field_constraint_function",
                              "no_field_constraints",
                              Patterns::Selection("no_field_constraints|norm_xphys_heaviside_gradients_constraint|norm_u_grad_xphys_constraint|abs_cos_u_grad_xphys_constraint|xphys_abs_cos_u_grad_xphys_constraint|fnorm_xphys_filter_hessians_constraint|norm_xphys_filter_gradients_constraint"),
                              "The dimensionless method (no_field_constraints|norm_xphys_heaviside_gradients_constraint|norm_u_grad_xphys_constraint|abs_cos_u_grad_xphys_constraint|xphys_abs_cos_u_grad_xphys_constraint|fnorm_xphys_filter_hessians_constraint|norm_xphys_filter_gradients_constraint)");
            prm.declare_entry("dimensionless_method",
                              "dimensionless_on",
                              Patterns::Selection("dimensionless_on|dimensionless_off"),
                              "The dimensionless method (dimensionless_on|dimensionless_off)");
            prm.declare_entry("normalization_method",
                              "normalization_on",
                              Patterns::Selection("normalization_on|normalization_off"),
                              "The normalization method (normalization_on|normalization_off)");
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization start parameters");
        {
            prm.declare_entry("start_point",
                              "0.75",
                              Patterns::Double(0, 1),
                              "Start with uniform fluid volume fraction");
            prm.declare_entry("obj_guess",
                              "1340",
                              Patterns::Double(0),
                              "The guess of objective function value");
            prm.declare_entry("vol_cstr",
                              "0.75",
                              Patterns::Double(0, 1),
                              "The maximum fluid volume fraction");
            prm.declare_entry("flow_cstr",
                              "1.0",
                              Patterns::Double(0),
                              "The maximum flow resistance value");
            prm.declare_entry("field_cstr",
                              "0.25",
                              Patterns::Double(0),
                              "The minimum field cooperation value");
            prm.declare_entry("field_if_epsilon",
                              "1e-8",
                              Patterns::Double(0),
                              "The field_if_epsilon value");
            prm.declare_entry("field_plus_epsilon",
                              "0",
                              Patterns::Double(0),
                              "The field_plus_epsilon value");

            prm.declare_entry("degree",
                              "1",
                              Patterns::Integer(0),
                              "The finite element order of physical field");
            prm.declare_entry("top_degree",
                              "2",
                              Patterns::Integer(0),
                              "The finite element order of density field");
            prm.declare_entry("loop",
                              "120",
                              Patterns::Integer(0),
                              "The iteration number of each mesh refinement");
            prm.declare_entry("refinement",
                              "0",
                              Patterns::Integer(0),
                              "The mesh refinement number");




        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization relax parameters");
        {
            prm.declare_entry("relax_heaviside_beta_step_init", "20", Patterns::Integer(0), "The initial step for relaxation (heaviside_beta)");
            prm.declare_entry("relax_heaviside_beta_step_size", "20", Patterns::Integer(0), "The step size for relaxation (heaviside_beta)");
            prm.declare_entry("relax_heaviside_beta_para_start", "1.0", Patterns::Double(0), "The start value for relaxation (heaviside_beta)");
            prm.declare_entry("relax_heaviside_beta_para_end", "8.0", Patterns::Double(0), "The end value for relaxation (heaviside_beta)");
            prm.declare_entry("relax_heaviside_beta_para_coeff", "2.0", Patterns::Double(0), "The relaxation coefficient (heaviside_beta)");

            prm.declare_entry("relax_heaviside_beta_equivalent_step_init", "80", Patterns::Integer(0), "The initial step for relaxation (heaviside_beta_equivalent)");
            prm.declare_entry("relax_heaviside_beta_equivalent_step_size", "20", Patterns::Integer(0), "The step size for relaxation (heaviside_beta_equivalent)");
            prm.declare_entry("relax_heaviside_beta_equivalent_para_start", "1.0", Patterns::Double(0), "The start value for relaxation (heaviside_beta_equivalent)");
            prm.declare_entry("relax_heaviside_beta_equivalent_para_end", "1.0", Patterns::Double(0), "The end value for relaxation (heaviside_beta_equivalent)");
            prm.declare_entry("relax_heaviside_beta_equivalent_para_coeff", "2.0", Patterns::Double(0), "The relaxation coefficient (heaviside_beta_equivalent)");

            prm.declare_entry("relax_da_refer_step_init", "20", Patterns::Integer(0), "The initial step for relaxation (Da_refer)");
            prm.declare_entry("relax_da_refer_step_size", "20", Patterns::Integer(0), "The step size for relaxation (Da_refer)");
            prm.declare_entry("relax_da_refer_para_start", "1e-3", Patterns::Double(0), "The start value for relaxation (Da_refer)");
            prm.declare_entry("relax_da_refer_para_end", "1e-4", Patterns::Double(0), "The end value for relaxation (Da_refer)");
            prm.declare_entry("relax_da_refer_para_coeff", "0.4", Patterns::Double(0), "The relaxation coefficient (Da_refer)");
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization filter and projection parameters");
        {
            prm.declare_entry("filter_radius_dml",
                              "0.2",
                              Patterns::Double(0),
                              "The dimensionless filter radius");
            prm.declare_entry("heaviside_yita",
                              "0.5",
                              Patterns::Double(0),
                              "The threshold of heaviside function");
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization boundary parameters");
        {

            prm.declare_entry("L_refer",
                              "2e-3",
                              Patterns::Double(0),
                              "The reference length (hydraulic diameter)");
            prm.declare_entry("U_refer",
                              "0.1",
                              Patterns::Double(0),
                              "The reference velocity (inlet velocity)");
            prm.declare_entry("rho_refer",
                              "1000.0",
                              Patterns::Double(0),
                              "The reference density (fluid)");
            prm.declare_entry("viscosity_refer",
                              "1e-3",
                              Patterns::Double(0),
                              "The reference viscosity (fluid)");

            prm.declare_entry("t_in_refer",
                              "10.0",
                              Patterns::Double(0),
                              "The reference inlet temperature");
            prm.declare_entry("t_Q_refer",
                              "30.0",
                              Patterns::Double(0),
                              "The reference heat source temperature");
            prm.declare_entry("capacity_refer",
                              "4183.0",
                              Patterns::Double(0),
                              "The reference capacity (fluid)");
            prm.declare_entry("k_f_refer",
                              "0.6",
                              Patterns::Double(0),
                              "The reference thermal conductivity (fluid)");
            prm.declare_entry("k_s_refer",
                              "180.0",
                              Patterns::Double(0),
                              "The reference thermal conductivity (solid)");
            prm.declare_entry("Q_refer",
                              "0.0",
                              Patterns::Double(0),
                              "The reference heat source intensity");
            prm.declare_entry("heatflux_refer",
                              "0.0",
                              Patterns::Double(0),
                              "The reference boundary heat flux");
            prm.declare_entry("heatsource_coeff_solid_dml",
                              "1e4",
                              Patterns::Double(0),
                              "The dimensionless heatsource coefficient (solid)");
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization mesh parameters");
        {
            prm.declare_entry("pre_refinement",
                              "0",
                              Patterns::Integer(0),
                              "Pre-refinement applied to the mesh before starting");
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization penalty parameters");
        {
            prm.declare_entry("permeability_penal",
                              "0.1",
                              Patterns::Double(0),
                              "The permeability penalty factor (RAMP)");
            prm.declare_entry("thermal_cond_penal",
                              "1.0",
                              Patterns::Double(0),
                              "The thermal conductivity penalty factor (RAMP)");
            prm.declare_entry("heatsource_coeff_penal",
                              "1.0",
                              Patterns::Double(0),
                              "The heatsource coefficient penalty factor (RAMP)");
            prm.declare_entry("convection_coeff_penal",
                              "1.0",
                              Patterns::Double(0),
                              "The convection coefficient penalty factor (RAMP)");
        }
        prm.leave_subsection();
    }

    void PRM::ParameterModifier::read_parameters(const std::string &parameter_file)
    {
        declare_parameters();
        prm.parse_input(parameter_file);
    }

    void PRM::ParameterModifier::write_parameters()
    {
        std::string selection_str;
        prm.enter_subsection("Topology optimization method parameters");
        {
            selection_str = prm.get("iteration_method");
            if (selection_str == std::string("newton_method"))
                iteration_method = newton_method;
            else if (selection_str == std::string("oseen_method"))
                iteration_method = oseen_method;
            else if (selection_str == std::string("stokes_method"))
                iteration_method = stokes_method;

            selection_str = prm.get("objective_function");
            if (selection_str == std::string("mean_temp_objective"))
                objective_function = mean_temp_objective;
            else if (selection_str == std::string("abs_u_gradt_objective"))
                objective_function = abs_u_gradt_objective;
            else if (selection_str == std::string("norm_gradt_objective"))
                objective_function = norm_gradt_objective;
            else if (selection_str == std::string("mean_solid_temp_objective"))
                objective_function = mean_solid_temp_objective;
            else if (selection_str == std::string("thermal_compliance_objective"))
                objective_function = thermal_compliance_objective;
            else if (selection_str == std::string("heat_exchange_objective"))
                objective_function = heat_exchange_objective;
            else if (selection_str == std::string("mean_boundary_temp_objective"))
                objective_function = mean_boundary_temp_objective;
            else if (selection_str == std::string("xphys_abs_cos_u_grad_temp_objective"))
                objective_function = xphys_abs_cos_u_grad_temp_objective;

            selection_str = prm.get("flow_constraint_function");
            if (selection_str == std::string("flow_dissipation_constraint"))
                flow_constraint_function = flow_dissipation_constraint;
            else if (selection_str == std::string("inlet_pressure_constraint"))
                flow_constraint_function = inlet_pressure_constraint;
            else if (selection_str == std::string("pump_power_constraint"))
                flow_constraint_function = pump_power_constraint;
            else if (selection_str == std::string("only_volume_constraint"))
                flow_constraint_function = only_volume_constraint;

            selection_str = prm.get("field_constraint_function");
            if (selection_str == std::string("no_field_constraints"))
                field_constraint_function = no_field_constraints;
            else if (selection_str == std::string("norm_xphys_heaviside_gradients_constraint"))
                field_constraint_function = norm_xphys_heaviside_gradients_constraint;
            else if (selection_str == std::string("norm_u_grad_xphys_constraint"))
                field_constraint_function = norm_u_grad_xphys_constraint;
            else if (selection_str == std::string("abs_cos_u_grad_xphys_constraint"))
                field_constraint_function = abs_cos_u_grad_xphys_constraint;
            else if (selection_str == std::string("xphys_abs_cos_u_grad_xphys_constraint"))
                field_constraint_function = xphys_abs_cos_u_grad_xphys_constraint;
            else if (selection_str == std::string("fnorm_xphys_filter_hessians_constraint"))
                field_constraint_function = fnorm_xphys_filter_hessians_constraint;
            else if (selection_str == std::string("norm_xphys_filter_gradients_constraint"))
                field_constraint_function = norm_xphys_filter_gradients_constraint;

            if ((prm.get("dimensionless_method")) == std::string("dimensionless_on"))
                PRM::dimensionless_method = PRM::dimensionless_on;
            else
                PRM::dimensionless_method = PRM::dimensionless_off;

            if ((prm.get("normalization_method")) == std::string("normalization_on"))
                PRM::normalization_method = PRM::normalization_on;
            else
                PRM::normalization_method = PRM::normalization_off;
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization start parameters");
        {
            PRM::start_point = prm.get_double("start_point");
            PRM::obj_guess = prm.get_double("obj_guess");
            PRM::vol_cstr = prm.get_double("vol_cstr");
            PRM::flow_cstr = prm.get_double("flow_cstr");
            PRM::field_cstr = prm.get_double("field_cstr");
            PRM::field_if_epsilon = prm.get_double("field_if_epsilon");
            PRM::field_plus_epsilon = prm.get_double("field_plus_epsilon");

            PRM::degree = prm.get_integer("degree");
            PRM::top_degree = prm.get_integer("top_degree");
            PRM::loop = prm.get_integer("loop");
            PRM::refinement = prm.get_integer("refinement");
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization relax parameters");
        {
            PRM::relax_heaviside_beta_step_init = prm.get_integer("relax_heaviside_beta_step_init");
            PRM::relax_heaviside_beta_step_size = prm.get_integer("relax_heaviside_beta_step_size");
            PRM::relax_heaviside_beta_para_start = prm.get_double("relax_heaviside_beta_para_start");
            PRM::relax_heaviside_beta_para_end = prm.get_double("relax_heaviside_beta_para_end");
            PRM::relax_heaviside_beta_para_coeff = prm.get_double("relax_heaviside_beta_para_coeff");

            PRM::relax_heaviside_beta_equivalent_step_init = prm.get_integer("relax_heaviside_beta_equivalent_step_init");
            PRM::relax_heaviside_beta_equivalent_step_size = prm.get_integer("relax_heaviside_beta_equivalent_step_size");
            PRM::relax_heaviside_beta_equivalent_para_start = prm.get_double("relax_heaviside_beta_equivalent_para_start");
            PRM::relax_heaviside_beta_equivalent_para_end = prm.get_double("relax_heaviside_beta_equivalent_para_end");
            PRM::relax_heaviside_beta_equivalent_para_coeff = prm.get_double("relax_heaviside_beta_equivalent_para_coeff");

            PRM::relax_da_refer_step_init = prm.get_integer("relax_da_refer_step_init");
            PRM::relax_da_refer_step_size = prm.get_integer("relax_da_refer_step_size");
            PRM::relax_da_refer_para_start = prm.get_double("relax_da_refer_para_start");
            PRM::relax_da_refer_para_end = prm.get_double("relax_da_refer_para_end");
            PRM::relax_da_refer_para_coeff = prm.get_double("relax_da_refer_para_coeff");
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization filter and projection parameters");
        {
            PRM::filter_radius_dml = prm.get_double("filter_radius_dml");
            PRM::heaviside_yita = prm.get_double("heaviside_yita");
            PRM::heaviside_beta = PRM::relax_heaviside_beta_para_start;
            PRM::heaviside_yita_equivalent = PRM::heaviside_yita;
            PRM::heaviside_beta_equivalent = relax_heaviside_beta_equivalent_para_start;
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization boundary parameters");
        {

            PRM::L_refer = prm.get_double("L_refer");
            PRM::U_refer = prm.get_double("U_refer");
            PRM::rho_refer = prm.get_double("rho_refer");
            PRM::viscosity_refer = prm.get_double("viscosity_refer");
            PRM::Re_refer = PRM::rho_refer * PRM::U_refer * PRM::L_refer / PRM::viscosity_refer;
            PRM::Da_refer = PRM::relax_da_refer_para_start;

            PRM::t_in_refer = prm.get_double("t_in_refer");
            PRM::t_Q_refer = prm.get_double("t_Q_refer");
            PRM::capacity_refer = prm.get_double("capacity_refer");
            PRM::k_f_refer = prm.get_double("k_f_refer");
            PRM::k_s_refer = prm.get_double("k_s_refer");
            PRM::Q_refer = prm.get_double("Q_refer");
            PRM::heatflux_refer = prm.get_double("heatflux_refer");
            PRM::heatsource_coeff_solid_dml = prm.get_double("heatsource_coeff_solid_dml");
            PRM::Pr_refer = PRM::viscosity_refer * PRM::capacity_refer / PRM::k_f_refer;
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization mesh parameters");
        {
            PRM::pre_refinement = prm.get_integer("pre_refinement");
        }
        prm.leave_subsection();

        prm.enter_subsection("Topology optimization penalty parameters");
        {
            PRM::permeability_fluid = 0.0;
            PRM::permeability_penal = prm.get_double("permeability_penal");
            PRM::thermal_cond_penal = prm.get_double("thermal_cond_penal");
            PRM::heatsource_coeff_penal = prm.get_double("heatsource_coeff_penal");
            PRM::convection_coeff_fluid = 1.0;
            PRM::convection_coeff_penal = prm.get_double("convection_coeff_penal");
        }
        prm.leave_subsection();

        switch (PRM::dimensionless_method)
        {
        case PRM::dimensionless_on:
        {
            /* Topology optimization filter and projection parameters */
            PRM::filter_radius = PRM::filter_radius_dml;

            /* Topology optimization boundary parameters */

            PRM::rho = 1.0;
            PRM::viscosity = 1.0 / PRM::Re_refer;
            PRM::capacity = PRM::Re_refer * PRM::Pr_refer;
            PRM::t_Q = 1.0;
            PRM::Q0 = PRM::L_refer * PRM::L_refer * PRM::Q_refer /
                      (PRM::k_f_refer * (PRM::t_Q_refer - PRM::t_in_refer));
            PRM::heatflux0 = PRM::L_refer * PRM::heatflux_refer /
                             (PRM::k_f_refer * (PRM::t_Q_refer - PRM::t_in_refer));
            PRM::inlet_velocity = 1.0;
            PRM::inlet_temperature = 0.0;

            /* Topology optimization material penalty parameters */
            PRM::permeability_solid = (1.0 + (1.0 / PRM::Re_refer)) / PRM::Da_refer;
            PRM::thermal_cond_fluid = 1.0;
            PRM::thermal_cond_solid = PRM::k_s_refer / PRM::k_f_refer;
            PRM::heatsource_coeff_solid = PRM::heatsource_coeff_solid_dml;
            break;
        }

        case PRM::dimensionless_off:
        {
            /* Topology optimization filter and projection parameters */
            PRM::filter_radius = PRM::filter_radius_dml * PRM::L_refer;

            /* Topology optimization boundary parameters */
            PRM::rho = PRM::rho_refer;
            PRM::viscosity = PRM::viscosity_refer;
            PRM::capacity = PRM::capacity_refer;
            PRM::t_Q = PRM::t_Q_refer;
            PRM::Q0 = PRM::Q_refer;
            PRM::heatflux0 = PRM::heatflux_refer;
            PRM::inlet_velocity = PRM::U_refer;
            PRM::inlet_temperature = PRM::t_in_refer;

            /* Topology optimization material penalty parameters */
            PRM::permeability_solid = (1.0 + (1.0 / PRM::Re_refer)) / PRM::Da_refer *
                                      PRM::rho_refer * PRM::U_refer / PRM::L_refer;
            PRM::thermal_cond_fluid = PRM::k_f_refer;
            PRM::thermal_cond_solid = PRM::k_s_refer;
            PRM::heatsource_coeff_solid = PRM::heatsource_coeff_solid_dml *
                                          PRM::k_f_refer / (PRM::L_refer * PRM::L_refer);

            break;
        }

        default:
        {
            break;
        }
        }
    }
}