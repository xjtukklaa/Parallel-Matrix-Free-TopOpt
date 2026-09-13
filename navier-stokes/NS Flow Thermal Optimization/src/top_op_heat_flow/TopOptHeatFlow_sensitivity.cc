#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /* 灵敏度组装——包括“等效变量 -> 投影变量 -> 过滤变量”过程，
    其中等效变量到投影变量的偏导由equivalent_partial实现，
    投影变量到过滤变量的偏导由heaviside_partial实现 */
    void TopOptHeatFlow::sens_assemble()
    {
        TimerOutput::Scope t(computing_timer, "sens_assembly");
        sens_obj_distributed = 0;
        sens_flow_distributed = 0;
        sens_vol_distributed = 0;
        sens_field_distributed = 0;

        QGauss<dim> quadrature_formula(quadrature_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values);
        FEValues<dim> temp_fe_values(temp_fe,
                                     quadrature_formula,
                                     update_values | update_gradients);
        FEValues<dim> ad_temp_fe_values(ad_temp_fe,
                                        quadrature_formula,
                                        update_values | update_gradients);
        FEValues<dim> ad_ns_fe_values(ad_ns_fe,
                                      quadrature_formula,
                                      update_values | update_gradients);
        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);
        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values | update_gradients | update_hessians | update_JxW_values);

        const unsigned int dofs_per_cell = top_dof_handler.get_fe_collection()[0].n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        const FEValuesExtractors::Vector ad_velocities(0);
        const FEValuesExtractors::Scalar ad_pressure(dim);

        Vector<double> local_sens_obj(dofs_per_cell);
        Vector<double> local_sens_flow(dofs_per_cell);
        Vector<double> local_sens_vol(dofs_per_cell);
        Vector<double> local_sens_field(dofs_per_cell);
        Vector<double> equivalent_partial(dofs_per_cell);
        Vector<double> heaviside_partial(dofs_per_cell);
        Vector<double> xphys_filter_dof_values(dofs_per_cell);
        double heavside_equivalent_partial;

        /*预分配内存*/
        std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
        std::vector<Tensor<1, dim>> present_temperature_gradients(n_q_points);
        std::vector<double> present_temperature_values(n_q_points);
        std::vector<Tensor<1, dim>> present_ad_temp_gradients(n_q_points);
        std::vector<double> present_ad_temp_values(n_q_points);
        std::vector<Tensor<1, dim>> present_ad_velocities_values(n_q_points);
        std::vector<Tensor<2, dim>> present_ad_velocities_gradients(n_q_points);
        std::vector<Tensor<1, dim>> present_ad_pressure_gradients(n_q_points);

        std::vector<Tensor<1, dim>> ns_flow_ad_velocities_values(n_q_points);
        std::vector<Tensor<2, dim>> ns_flow_ad_velocities_gradients(n_q_points);
        std::vector<Tensor<1, dim>> ns_flow_ad_pressure_gradients(n_q_points);

        std::vector<Tensor<1, dim>> ns_field_ad_velocities_values(n_q_points);
        std::vector<Tensor<2, dim>> ns_field_ad_velocities_gradients(n_q_points);
        std::vector<Tensor<1, dim>> ns_field_ad_pressure_gradients(n_q_points);

        std::vector<double> xphys_heaviside_values(n_q_points);
        std::vector<Tensor<1, dim>> xphys_heaviside_gradients(n_q_points);
        std::vector<Tensor<1, dim>> xphys_filter_gradients(n_q_points);
        std::vector<Tensor<2, dim>> xphys_filter_hessians(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);

        std::vector<double> phi_top(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_top(dofs_per_cell);
        std::vector<Tensor<2, dim>> hessian_phi_top(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*SUPG-PSPG修正参数*/
        std::vector<double> tauSUPG(n_q_points), tauPSPG(n_q_points), vLSIC(n_q_points), tauSUPGT(n_q_points);
        StabilizationParameter<dim> stabilization_parameter;
        double kinematic_viscosity = viscosity / rho;
        const double thermal_diffusivity = thermal_cond_fluid / (rho * capacity);

        double volume_total = GridTools::volume(triangulation);
        double volume_design = domain_volume[0];

        /*迭代器*/
        auto cell = top_dof_handler.begin_active();
        const auto endc = top_dof_handler.end();
        auto ns_cell = ns_dof_handler.begin_active();
        auto temp_cell = temp_dof_handler.begin_active();
        auto ad_temp_cell = ad_temp_dof_handler.begin_active();
        auto ad_ns_cell = ad_ns_dof_handler.begin_active();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++ns_cell, ++temp_cell, ++ad_temp_cell, ++ad_ns_cell, ++top_equivalent_cell)
        {

            if (cell->is_locally_owned() && cell->material_id() == 0)
            {
                local_sens_obj = 0;
                local_sens_flow = 0;
                local_sens_vol = 0;
                local_sens_field = 0;

                equivalent_partial = 0;
                heaviside_partial = 0;
                heavside_equivalent_partial = 0;
                double local_volume = cell->measure();
                top_fe_values.reinit(cell);
                ns_fe_values.reinit(ns_cell);
                temp_fe_values.reinit(temp_cell);
                ad_temp_fe_values.reinit(ad_temp_cell);
                ad_ns_fe_values.reinit(ad_ns_cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                ns_fe_values[velocities].get_function_values(ns_solution_relevant,
                                                             present_velocity_values);
                temp_fe_values.get_function_gradients(temp_solution_relevant,
                                                      present_temperature_gradients);
                temp_fe_values.get_function_values(temp_solution_relevant,
                                                   present_temperature_values);
                ad_temp_fe_values.get_function_gradients(ad_temp_solution_relevant,
                                                         present_ad_temp_gradients);
                ad_temp_fe_values.get_function_values(ad_temp_solution_relevant,
                                                      present_ad_temp_values);
                ad_ns_fe_values[ad_velocities].get_function_values(ad_ns_solution_relevant,
                                                                   present_ad_velocities_values);
                ad_ns_fe_values[ad_velocities].get_function_gradients(ad_ns_solution_relevant,
                                                                      present_ad_velocities_gradients);
                ad_ns_fe_values[ad_pressure].get_function_gradients(ad_ns_solution_relevant,
                                                                    present_ad_pressure_gradients);

                ad_ns_fe_values[ad_velocities].get_function_values(ad_ns_flow_solution_relevant,
                                                                   ns_flow_ad_velocities_values);
                ad_ns_fe_values[ad_velocities].get_function_gradients(ad_ns_flow_solution_relevant,
                                                                      ns_flow_ad_velocities_gradients);
                ad_ns_fe_values[ad_pressure].get_function_gradients(ad_ns_flow_solution_relevant,
                                                                    ns_flow_ad_pressure_gradients);

                ad_ns_fe_values[ad_velocities].get_function_values(ad_ns_field_solution_relevant,
                                                                   ns_field_ad_velocities_values);
                ad_ns_fe_values[ad_velocities].get_function_gradients(ad_ns_field_solution_relevant,
                                                                      ns_field_ad_velocities_gradients);
                ad_ns_fe_values[ad_pressure].get_function_gradients(ad_ns_field_solution_relevant,
                                                                    ns_field_ad_pressure_gradients);

                auto &present_top_fe_values = top_fe_values.get_present_fe_values();
                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_fe_values.get_function_values(xphys_heaviside_relevant,
                                                          xphys_heaviside_values);
                present_top_fe_values.get_function_gradients(xphys_heaviside_relevant,
                                                             xphys_heaviside_gradients);
                present_top_fe_values.get_function_gradients(xphys_filter_relevant,
                                                             xphys_filter_gradients);
                present_top_fe_values.get_function_hessians(xphys_filter_relevant,
                                                            xphys_filter_hessians);
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);

                cell->get_dof_values(xphys_filter_relevant, xphys_filter_dof_values);

                double element_size = (cell->diameter()) / std::sqrt(2);

                stabilization_parameter.value(present_velocity_values,
                                              kinematic_viscosity, element_size,
                                              tauSUPG, tauPSPG, vLSIC);
                stabilization_parameter.value_temperature(present_velocity_values,
                                                          thermal_diffusivity, element_size, tauSUPGT);

                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    heaviside_partial[k] = heaviside.partial(xphys_filter_dof_values(k));
                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        phi_top[k] = present_top_fe_values.shape_value(k, q);
                        equivalent_partial[k] += 1 / local_volume * phi_top[k] * present_top_fe_values.JxW(q);
                    }
                }

                double xphys_heaviside_average = 0.0;
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    xphys_heaviside_average += 1 / local_volume * xphys_heaviside_values[q] * present_top_fe_values.JxW(q);
                }
                heavside_equivalent_partial = heaviside_equivalent.partial(xphys_heaviside_average);

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);
                    double alpha_partial =
                        material_interpolate.permeability_partial(xphys_equivalent_values_q_point);
                    double kcond_partial =
                        material_interpolate.conductivity_partial(xphys_equivalent_values_q_point);
                    double heatsource_coeff_partial =
                        material_interpolate.heatsource_partial(xphys_equivalent_values_q_point);
                    double convection_coeff_partial =
                        material_interpolate.convection_partial(xphys_equivalent_values_q_point);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        phi_top[k] = present_top_fe_values.shape_value(k, q);
                        grad_phi_top[k] = present_top_fe_values.shape_grad(k, q);
                        hessian_phi_top[k] = present_top_fe_values.shape_hessian(k, q);
                    }

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        /* 注意，以local_sens_obj为例，其中除了equivalent_partial[k]之外的项与k无关，
                        其实可以把无关的量提取到k的循环的外侧，作为一个定值，赋值给ele_sens_obj，
                        但现在这种布置方式代码更整洁，虽然浪费了计算资源。*/

                        local_sens_obj[k] +=
                            ((present_ad_velocities_values[q] * present_velocity_values[q] * alpha_partial) +
                             (tauSUPG[q] * (present_ad_velocities_gradients[q] * present_velocity_values[q]) *
                              present_velocity_values[q] * alpha_partial) +
                             ((1.0 / rho) * tauPSPG[q] *
                              (present_ad_pressure_gradients[q] * present_velocity_values[q]) * alpha_partial) +
                             (present_ad_temp_values[q] *
                              (rho * capacity * (present_velocity_values[q] * present_temperature_gradients[q]) * convection_coeff_partial -
                               (t_Q - present_temperature_values[q]) * heatsource_coeff_partial)) +
                             ((present_ad_temp_gradients[q] * present_temperature_gradients[q]) * kcond_partial) +
                             ((tauSUPGT[q] * (present_velocity_values[q] * present_ad_temp_gradients[q])) *
                              (rho * capacity * (present_velocity_values[q] * present_temperature_gradients[q]) * convection_coeff_partial -
                               (t_Q - present_temperature_values[q]) * heatsource_coeff_partial))) *
                            heavside_equivalent_partial * equivalent_partial[k] * heaviside_partial[k] *
                            present_top_fe_values.JxW(q);

                        switch (objective_function)
                        {
                        case PRM::mean_temp_objective:
                            break;

                        case PRM::abs_u_gradt_objective:
                            break;

                        case PRM::norm_gradt_objective:
                            break;

                        case PRM::mean_solid_temp_objective:
                        {
                            local_sens_obj[k] +=
                                ((-1.0) / volume_total * 1.0) *
                                heavside_equivalent_partial * equivalent_partial[k] * heaviside_partial[k] *
                                present_top_fe_values.JxW(q);
                            break;
                        }

                        case PRM::thermal_compliance_objective:
                        {
                            local_sens_obj(k) +=
                                (kcond_partial * present_temperature_gradients[q].norm_square()) *
                                heavside_equivalent_partial * equivalent_partial[k] * heaviside_partial[k] *
                                present_top_fe_values.JxW(q);
                            break;
                        }

                        case PRM::heat_exchange_objective:
                        {
                            local_sens_obj(k) +=
                                (-1.0) *
                                (t_Q - present_temperature_values[q]) * heatsource_coeff_partial *
                                heavside_equivalent_partial * equivalent_partial[k] * heaviside_partial[k] *
                                present_top_fe_values.JxW(q);
                            break;
                        }
                        case PRM::mean_boundary_temp_objective:
                            break;

                        case PRM::xphys_abs_cos_u_grad_temp_objective:
                        {
                            if (std::abs(present_velocity_values[q] * present_temperature_gradients[q]) > field_if_epsilon)
                            {
                                local_sens_obj(k) +=
                                    (-1.0) *
                                    ((std::abs(present_velocity_values[q] * present_temperature_gradients[q]) /
                                      (present_velocity_values[q].norm() * present_temperature_gradients[q].norm() + field_plus_epsilon)) /
                                     volume_total) *
                                    heavside_equivalent_partial * equivalent_partial[k] * heaviside_partial[k] *
                                    present_top_fe_values.JxW(q);
                            }
                            break;
                        }

                        default:
                            break;
                        }

                        local_sens_vol(k) +=
                            1.0 / volume_design *
                            phi_top[k] * heaviside_partial[k] *
                            present_top_fe_values.JxW(q);

                        /* 约束函数-默认不包含伪密度插值项 */

                        local_sens_flow(k) +=
                            ((ns_flow_ad_velocities_values[q] * present_velocity_values[q] * alpha_partial) +
                             (tauSUPG[q] * (ns_flow_ad_velocities_gradients[q] * present_velocity_values[q]) *
                              present_velocity_values[q] * alpha_partial) +
                             ((1.0 / rho) * tauPSPG[q] *
                              (ns_flow_ad_pressure_gradients[q] * present_velocity_values[q]) * alpha_partial)) *
                            heavside_equivalent_partial * equivalent_partial[k] * heaviside_partial[k] *
                            present_top_fe_values.JxW(q);

                        switch (flow_constraint_function)
                        {
                        case PRM::flow_dissipation_constraint:
                        {
                            local_sens_flow(k) +=
                                (alpha_partial * present_velocity_values[q].norm_square()) *
                                heavside_equivalent_partial * equivalent_partial[k] * heaviside_partial[k] *
                                present_top_fe_values.JxW(q);
                            break;
                        }

                        case PRM::inlet_pressure_constraint:
                            break;

                        case PRM::pump_power_constraint:
                            break;

                        case PRM::only_volume_constraint:
                        {
                            local_sens_flow(k) = local_sens_vol(k);
                            break;
                        }

                        default:
                            break;
                        }

                        /* 约束函数-默认不包含伪密度插值项 */

                        local_sens_field(k) +=
                            ((ns_field_ad_velocities_values[q] * present_velocity_values[q] * alpha_partial) +
                             (tauSUPG[q] * (ns_field_ad_velocities_gradients[q] * present_velocity_values[q]) *
                              present_velocity_values[q] * alpha_partial) +
                             ((1.0 / rho) * tauPSPG[q] *
                              (ns_field_ad_pressure_gradients[q] * present_velocity_values[q]) * alpha_partial)) *
                            heavside_equivalent_partial * equivalent_partial[k] * heaviside_partial[k] *
                            present_top_fe_values.JxW(q);

                        switch (field_constraint_function)
                        {
                        case PRM::no_field_constraints:
                            break;

                        case PRM::norm_xphys_heaviside_gradients_constraint:
                        {
                            if (xphys_heaviside_gradients[q].norm() > field_if_epsilon)
                            {
                                local_sens_field(k) +=
                                    (1.0) *
                                    (xphys_heaviside_gradients[q] * grad_phi_top[k]) /
                                    (xphys_heaviside_gradients[q].norm() + field_plus_epsilon) *
                                    (1 / volume_total) *
                                    heaviside_partial[k] *
                                    present_top_fe_values.JxW(q);
                            }
                            break;
                        }

                        case PRM::abs_cos_u_grad_xphys_constraint:
                        {
                            if (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) > field_if_epsilon)
                            {
                                local_sens_field(k) +=
                                    (-1.0) *
                                    ((grad_phi_top[k]) *
                                     ((((present_velocity_values[q] * xphys_heaviside_gradients[q]) /
                                        (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) + field_plus_epsilon) *
                                        present_velocity_values[q].norm() *
                                        xphys_heaviside_gradients[q].norm()) *
                                       present_velocity_values[q]) -
                                      ((std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) *
                                        present_velocity_values[q].norm() / (xphys_heaviside_gradients[q].norm() + field_plus_epsilon)) *
                                       xphys_heaviside_gradients[q])) /
                                     (present_velocity_values[q].norm_square() * xphys_heaviside_gradients[q].norm_square() + field_plus_epsilon) /
                                     volume_total) *
                                    heaviside_partial[k] *
                                    present_top_fe_values.JxW(q);
                            }

                            break;
                        }

                        case PRM::xphys_abs_cos_u_grad_xphys_constraint:
                        {
                            if (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) > field_if_epsilon)
                            {
                                local_sens_field(k) +=
                                    (-1.0) *
                                    ((((grad_phi_top[k]) *
                                       ((((present_velocity_values[q] * xphys_heaviside_gradients[q]) /
                                          (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) + field_plus_epsilon) *
                                          present_velocity_values[q].norm() *
                                          xphys_heaviside_gradients[q].norm()) *
                                         present_velocity_values[q]) -
                                        ((std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) *
                                          present_velocity_values[q].norm() / (xphys_heaviside_gradients[q].norm() + field_plus_epsilon)) *
                                         xphys_heaviside_gradients[q])) /
                                       (present_velocity_values[q].norm_square() * xphys_heaviside_gradients[q].norm_square() + field_plus_epsilon)) +
                                      (phi_top[k] * std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) /
                                       (present_velocity_values[q].norm() * xphys_heaviside_gradients[q].norm() + field_plus_epsilon))) /
                                     volume_total) *
                                    heaviside_partial[k] *
                                    present_top_fe_values.JxW(q);
                            }

                            break;
                        }

                        case PRM::norm_u_grad_xphys_constraint:
                        {
                            if (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) > field_if_epsilon)
                            {
                                local_sens_field(k) +=
                                    (-1.0) *
                                    ((grad_phi_top[k]) *
                                     ((((present_velocity_values[q] * xphys_heaviside_gradients[q]) /
                                        (std::abs(present_velocity_values[q] * xphys_heaviside_gradients[q]) + field_plus_epsilon)) *
                                       present_velocity_values[q])) /
                                     volume_total) *
                                    heaviside_partial[k] *
                                    present_top_fe_values.JxW(q);
                            }
                            break;
                        }

                        case PRM::fnorm_xphys_filter_hessians_constraint:
                        {
                            if (xphys_filter_hessians[q].norm() > field_if_epsilon)
                            {
                                local_sens_field(k) +=
                                    (-1.0) *
                                    scalar_product(xphys_filter_hessians[q], hessian_phi_top[k]) /
                                    (xphys_filter_hessians[q].norm() + field_plus_epsilon) *
                                    (1 / volume_total) *
                                    heaviside_partial[k] *
                                    present_top_fe_values.JxW(q);
                            }
                            break;
                        }

                        case PRM::norm_xphys_filter_gradients_constraint:
                        {
                            if (xphys_filter_gradients[q].norm() > field_if_epsilon)
                            {
                                local_sens_field(k) +=
                                    (1.0) *
                                    (xphys_filter_gradients[q] * grad_phi_top[k]) /
                                    (xphys_filter_gradients[q].norm() + field_plus_epsilon) *
                                    (1 / volume_total) *
                                    heaviside_partial[k] *
                                    present_top_fe_values.JxW(q);
                            }
                            break;
                        }

                        default:
                            break;
                        }
                    }
                }

                if (top_degree == 0)
                {
                    cell->set_dof_values(local_sens_obj, sens_obj_distributed);
                    cell->set_dof_values(local_sens_flow, sens_flow_distributed);
                    cell->set_dof_values(local_sens_vol, sens_vol_distributed);
                    cell->set_dof_values(local_sens_field, sens_field_distributed);
                }
                else
                {
                    cell->get_dof_indices(local_dof_indices);

                    top_constraints.distribute_local_to_global(local_sens_obj,
                                                               local_dof_indices,
                                                               sens_obj_distributed);
                    top_constraints.distribute_local_to_global(local_sens_flow,
                                                               local_dof_indices,
                                                               sens_flow_distributed);
                    top_constraints.distribute_local_to_global(local_sens_vol,
                                                               local_dof_indices,
                                                               sens_vol_distributed);
                    top_constraints.distribute_local_to_global(local_sens_field,
                                                               local_dof_indices,
                                                               sens_field_distributed);
                }
            }
        }

        if (top_degree == 0)
        {
            sens_obj_distributed.compress(VectorOperation::insert);
            sens_flow_distributed.compress(VectorOperation::insert);
            sens_vol_distributed.compress(VectorOperation::insert);
            sens_field_distributed.compress(VectorOperation::insert);
        }
        else
        {
            sens_obj_distributed.compress(VectorOperation::add);
            sens_flow_distributed.compress(VectorOperation::add);
            sens_vol_distributed.compress(VectorOperation::add);
            sens_field_distributed.compress(VectorOperation::add);
        }

        top_constraints.distribute(sens_obj_distributed);
        top_constraints.distribute(sens_flow_distributed);
        top_constraints.distribute(sens_vol_distributed);
        top_constraints.distribute(sens_field_distributed);

        sens_obj_relevant = sens_obj_distributed;
        sens_flow_relevant = sens_flow_distributed;
        sens_vol_relevant = sens_vol_distributed;
        sens_field_relevant = sens_field_distributed;
    }

    /* 该函数已弃用————目标函数对xphys_heav的导数 * xphys_heav对xphys_filter的偏导 */

    void TopOptHeatFlow::sensitivity_dheaviside()
    {
        TimerOutput::Scope t(computing_timer, "sensitivity_dheaviside");

        sens_obj_dheaviside_distributed = 0;
        sens_flow_dheaviside_distributed = 0;
        sens_vol_dheaviside_distributed = 0;
        sens_field_dheaviside_distributed = 0;

        QGauss<dim> quadrature_formula(top_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values);
        const unsigned int dofs_per_cell = top_dof_handler.get_fe_collection()[0].n_dofs_per_cell();

        Vector<double> local_sens_obj_values(dofs_per_cell);
        Vector<double> local_sens_flow_values(dofs_per_cell);
        Vector<double> local_sens_vol_values(dofs_per_cell);
        Vector<double> local_sens_field_values(dofs_per_cell);

        /*预分配内存*/
        Vector<double> xphys_filter_dof_values(dofs_per_cell);
        Vector<double> sens_obj_dof_values(dofs_per_cell);
        Vector<double> sens_flow_dof_values(dofs_per_cell);
        Vector<double> sens_vol_dof_values(dofs_per_cell);
        Vector<double> sens_field_dof_values(dofs_per_cell);

        /*迭代器*/
        auto cell = top_dof_handler.begin_active();
        const auto endc = top_dof_handler.end();

        for (; cell != endc; ++cell)
        {

            if (cell->is_locally_owned() && cell->material_id() == 0)
            {
                local_sens_obj_values = 0.0;
                local_sens_flow_values = 0.0;
                local_sens_vol_values = 0.0;
                local_sens_field_values = 0.0;

                top_fe_values.reinit(cell);

                cell->get_dof_values(xphys_filter_relevant, xphys_filter_dof_values);
                cell->get_dof_values(sens_obj_relevant, sens_obj_dof_values);
                cell->get_dof_values(sens_flow_relevant, sens_flow_dof_values);
                cell->get_dof_values(sens_vol_relevant, sens_vol_dof_values);
                cell->get_dof_values(sens_field_relevant, sens_field_dof_values);

                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    double dheav = heaviside.partial(xphys_filter_dof_values(k));
                    local_sens_obj_values(k) =
                        sens_obj_dof_values(k) * dheav;
                    local_sens_flow_values(k) =
                        sens_flow_dof_values(k) * dheav;
                    local_sens_vol_values(k) =
                        sens_vol_dof_values(k) * dheav;
                    local_sens_field_values(k) =
                        sens_field_dof_values(k) * dheav;
                }

                cell->set_dof_values(local_sens_obj_values, sens_obj_dheaviside_distributed);
                cell->set_dof_values(local_sens_flow_values, sens_flow_dheaviside_distributed);
                cell->set_dof_values(local_sens_vol_values, sens_vol_dheaviside_distributed);
                cell->set_dof_values(local_sens_field_values, sens_field_dheaviside_distributed);
            }
        }

        sens_obj_dheaviside_distributed.compress(VectorOperation::insert);
        sens_flow_dheaviside_distributed.compress(VectorOperation::insert);
        sens_vol_dheaviside_distributed.compress(VectorOperation::insert);
        sens_field_dheaviside_distributed.compress(VectorOperation::insert);

        top_constraints.distribute(sens_obj_dheaviside_distributed);
        top_constraints.distribute(sens_flow_dheaviside_distributed);
        top_constraints.distribute(sens_vol_dheaviside_distributed);
        top_constraints.distribute(sens_field_dheaviside_distributed);

        sens_obj_dheaviside_relevant = sens_obj_dheaviside_distributed;
        sens_flow_dheaviside_relevant = sens_flow_dheaviside_distributed;
        sens_vol_dheaviside_relevant = sens_vol_dheaviside_distributed;
        sens_field_dheaviside_relevant = sens_field_dheaviside_distributed;
    }

}