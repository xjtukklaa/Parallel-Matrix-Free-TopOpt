#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*连续伴随NS组装*/
    void TopOptHeatFlow::ad_continuous_ns_assemble(const PRM::LagrangeFunctionType lag_function, const bool assemble_matrix)
    {
        TimerOutput::Scope t(computing_timer, "ad_ns_assembly");

        if (assemble_matrix)
        {
            ad_ns_system_matrix = 0;
        }

        ad_ns_system_rhs = 0;

        QGauss<dim> quadrature_formula(quadrature_degree + 1);
        QGauss<dim - 1> face_quadrature_formula(quadrature_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values | update_gradients);
        FEValues<dim> temp_fe_values(temp_fe,
                                     quadrature_formula,
                                     update_values | update_gradients);
        FEValues<dim> ad_temp_fe_values(ad_temp_fe,
                                        quadrature_formula,
                                        update_values | update_gradients);
        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values | update_gradients);
        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);
        FEValues<dim> ad_ns_fe_values(ad_ns_fe,
                                      quadrature_formula,
                                      update_values | update_quadrature_points |
                                          update_JxW_values | update_gradients);

        FEFaceValues<dim> ns_fe_face_values(ns_fe,
                                            face_quadrature_formula,
                                            update_values);
        FEFaceValues<dim> fe_face_values(ad_ns_fe,
                                         face_quadrature_formula,
                                         update_values | update_normal_vectors |
                                             update_JxW_values);

        const unsigned int dofs_per_cell = ad_ns_fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        const FEValuesExtractors::Vector ad_velocities(0);
        const FEValuesExtractors::Scalar ad_pressure(dim);

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*温度方程右端项-热源项*/
        TempRightHandSide<dim> temp_rhs;
        double volume_total = GridTools::volume(triangulation);

        /*预分配内存*/
        std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
        std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
        std::vector<double> present_velocity_divergences(n_q_points);
        std::vector<Tensor<1, dim>> present_pressure_gradients(n_q_points);
        std::vector<Tensor<1, dim>> present_temperature_gradients(n_q_points);
        std::vector<double> present_temperature_values(n_q_points);

        std::vector<double> present_ad_temp_values(n_q_points);
        std::vector<Tensor<1, dim>> present_ad_temp_gradients(n_q_points);
        std::vector<double> xphys_values(n_q_points);
        std::vector<Tensor<1, dim>> xphys_gradients(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);
        std::vector<double> temp_rhs_values(n_q_points);

        std::vector<double> present_face_pressure_values(n_face_q_points);
        std::vector<Tensor<1, dim>> present_face_velocity_values(n_face_q_points);
        std::vector<Tensor<1, dim>> present_face_normal_vectors(n_face_q_points);

        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>> symmgrad_phi_u(dofs_per_cell);
        std::vector<double> phi_p(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

        /*SUPG-PSPG-LSIC稳定化参数及其对速度矢量的偏导*/
        std::vector<double> tauSUPG(n_q_points), tauPSPG(n_q_points), vLSIC(n_q_points), tauSUPGT(n_q_points);
        std::vector<Tensor<1, dim>> delta_tauSUPG(n_q_points), delta_tauPSPG(n_q_points),
            delta_vLSIC(n_q_points), delta_tauSUPGT(n_q_points);
        StabilizationParameter<dim> stabilization_parameter;
        double kinematic_viscosity = viscosity / rho;
        const double thermal_diffusivity = thermal_cond_fluid / (rho * capacity);

        /*迭代器*/
        auto cell = ad_ns_dof_handler.begin_active();
        const auto endc = ad_ns_dof_handler.end();
        auto ns_cell = ns_dof_handler.begin_active();
        auto temp_cell = temp_dof_handler.begin_active();
        auto ad_temp_cell = ad_temp_dof_handler.begin_active();
        auto top_cell = top_dof_handler.begin_active();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++ns_cell, ++temp_cell, ++ad_temp_cell, ++top_cell, ++top_equivalent_cell)
        {
            if (cell->is_locally_owned())
            {
                ad_ns_fe_values.reinit(cell);
                ns_fe_values.reinit(ns_cell);
                temp_fe_values.reinit(temp_cell);
                ad_temp_fe_values.reinit(ad_temp_cell);
                top_fe_values.reinit(top_cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                local_matrix = 0;
                local_rhs = 0;

                /*读取高斯点处的函数值*/
                ns_fe_values[velocities].get_function_values(ns_solution_relevant,
                                                             present_velocity_values);
                ns_fe_values[velocities].get_function_gradients(ns_solution_relevant,
                                                                present_velocity_gradients);
                ns_fe_values[velocities].get_function_divergences(ns_solution_relevant,
                                                                  present_velocity_divergences);
                ns_fe_values[pressure].get_function_gradients(ns_solution_relevant,
                                                              present_pressure_gradients);
                temp_fe_values.get_function_gradients(temp_solution_relevant,
                                                      present_temperature_gradients);
                temp_fe_values.get_function_values(temp_solution_relevant,
                                                   present_temperature_values);
                ad_temp_fe_values.get_function_values(ad_temp_solution_relevant,
                                                      present_ad_temp_values);
                ad_temp_fe_values.get_function_gradients(ad_temp_solution_relevant,
                                                         present_ad_temp_gradients);
                auto &present_top_fe_values = top_fe_values.get_present_fe_values();
                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_fe_values.get_function_values(xphys_heaviside_relevant,
                                                          xphys_values);
                present_top_fe_values.get_function_gradients(xphys_heaviside_relevant,
                                                             xphys_gradients);
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);

                double element_size = (cell->diameter()) / std::sqrt(2);

                stabilization_parameter.value(present_velocity_values,
                                              kinematic_viscosity, element_size,
                                              tauSUPG, tauPSPG, vLSIC);

                stabilization_parameter.value_temperature(present_velocity_values,
                                                          thermal_diffusivity, element_size, tauSUPGT);
                stabilization_parameter.partial(present_velocity_values,
                                                kinematic_viscosity, thermal_diffusivity, element_size,
                                                delta_tauSUPG, delta_tauPSPG,
                                                delta_vLSIC, delta_tauSUPGT);

                /*SUPG-PSPG-LSIC修正  局部矩阵组装*/
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);
                    temp_rhs_values[q] = temp_rhs.value(ad_ns_fe_values.quadrature_point(q));
                    permeability = material_interpolate.permeability_interpolate(xphys_equivalent_values_q_point);
                    thermal_conductivity = material_interpolate.conductivity_interpolate(xphys_equivalent_values_q_point);
                    heatsource_coeff = material_interpolate.heatsource_interpolate(xphys_equivalent_values_q_point);
                    convection_coeff = material_interpolate.convection_interpolate(xphys_equivalent_values_q_point);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        div_phi_u[k] = ad_ns_fe_values[ad_velocities].divergence(k, q);
                        grad_phi_u[k] = ad_ns_fe_values[ad_velocities].gradient(k, q);
                        symmgrad_phi_u[k] = ad_ns_fe_values[ad_velocities].symmetric_gradient(k, q);
                        phi_u[k] = ad_ns_fe_values[ad_velocities].value(k, q);
                        phi_p[k] = ad_ns_fe_values[ad_pressure].value(k, q);
                        grad_phi_p[k] = ad_ns_fe_values[ad_pressure].gradient(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        if (assemble_matrix)
                        {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                local_matrix(i, j) +=
                                    ((phi_u[j]) *
                                         ((rho * ((present_velocity_gradients[q] * phi_u[i]) + (grad_phi_u[i] * present_velocity_values[q]))) +
                                          (permeability * phi_u[i])) +
                                     (2 * viscosity * scalar_product(grad_phi_u[j], symmgrad_phi_u[i])) -
                                     (div_phi_u[j] * phi_p[i]) +
                                     (phi_p[j] * div_phi_u[i]) +

                                     ((tauSUPG[q] * (grad_phi_u[j] * present_velocity_values[q])) + ((1.0 / rho) * tauPSPG[q] * grad_phi_p[j])) *
                                         ((rho * ((present_velocity_gradients[q] * phi_u[i]) + (grad_phi_u[i] * present_velocity_values[q]))) +
                                          grad_phi_p[i] +
                                          (permeability * phi_u[i])) +

                                     ((tauSUPG[q] * (grad_phi_u[j] * phi_u[i])) +
                                      (delta_tauSUPG[q] * phi_u[i]) * (grad_phi_u[j] * present_velocity_values[q]) +
                                      ((1.0 / rho) * (delta_tauPSPG[q] * phi_u[i]) * grad_phi_p[j])) *
                                         ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                          present_pressure_gradients[q] +
                                          (permeability * present_velocity_values[q])) +

                                     (vLSIC[q] * rho * div_phi_u[i] * div_phi_u[j]) +
                                     ((delta_vLSIC[q] * phi_u[i]) * rho * div_phi_u[j] * present_velocity_divergences[q])) *
                                    ad_ns_fe_values.JxW(q);
                            }
                        }

                        switch (lag_function)
                        {
                        case PRM::lagrange_obj_function:
                        {
                            local_rhs(i) -=
                                ((present_ad_temp_values[q] + tauSUPGT[q] * (present_velocity_values[q] * present_ad_temp_gradients[q])) *
                                     (convection_coeff * rho * capacity * (phi_u[i] * present_temperature_gradients[q])) +

                                 (tauSUPGT[q] * (phi_u[i] * present_ad_temp_gradients[q]) +
                                  (delta_tauSUPGT[q] * phi_u[i]) * (present_velocity_values[q] * present_ad_temp_gradients[q])) *
                                     (convection_coeff * rho * capacity * (present_velocity_values[q] * present_temperature_gradients[q]) -
                                      temp_rhs_values[q] -
                                      heatsource_coeff * (t_Q - present_temperature_values[q]))) *
                                ad_ns_fe_values.JxW(q);

                            switch (objective_function)
                            {
                            case PRM::mean_temp_objective:
                                break;

                            case PRM::abs_u_gradt_objective:
                            {
                                local_rhs(i) -=
                                    (-1.0) *
                                    ((phi_u[i] * present_temperature_gradients[q]) *
                                     (present_velocity_values[q] * present_temperature_gradients[q]) /
                                     std::abs(present_velocity_values[q] * present_temperature_gradients[q])) *
                                    ad_ns_fe_values.JxW(q);
                                break;
                            }

                            case PRM::norm_gradt_objective:
                                break;

                            case PRM::mean_solid_temp_objective:
                                break;

                            case PRM::thermal_compliance_objective:
                            {
                                local_rhs(i) -=
                                    ((phi_u[i] * present_temperature_gradients[q]) * rho * capacity * present_temperature_values[q]) *
                                    ad_ns_fe_values.JxW(q);
                                break;
                            }

                            case PRM::heat_exchange_objective:
                                break;

                            case PRM::mean_boundary_temp_objective:
                                break;

                            case PRM::xphys_abs_cos_u_grad_temp_objective:
                            {
                                if (std::abs(present_velocity_values[q] * present_temperature_gradients[q]) > field_if_epsilon)
                                {
                                    local_rhs(i) -=
                                        (-1.0) *
                                        ((phi_u[i]) *
                                         ((((present_velocity_values[q] * present_temperature_gradients[q]) /
                                            (std::abs(present_velocity_values[q] * present_temperature_gradients[q]) + field_plus_epsilon) *
                                            present_velocity_values[q].norm() * present_temperature_gradients[q].norm()) *
                                           present_temperature_gradients[q]) -
                                          ((std::abs(present_velocity_values[q] * present_temperature_gradients[q]) *
                                            present_temperature_gradients[q].norm() / present_velocity_values[q].norm()) *
                                           present_velocity_values[q])) /
                                         (present_velocity_values[q].norm_square() * present_temperature_gradients[q].norm_square() + field_plus_epsilon) *
                                         xphys_equivalent_values[q] /
                                         volume_total) *
                                        ad_ns_fe_values.JxW(q);
                                }
                                break;
                            }

                            default:
                                break;
                            }

                            break;
                        }

                        case PRM::lagrange_flow_cstr_function:
                        {
                            switch (flow_constraint_function)
                            {
                            case PRM::flow_dissipation_constraint:
                            {
                                local_rhs(i) -=
                                    ((viscosity * scalar_product(grad_phi_u[i], present_velocity_gradients[q])) +
                                     ((phi_u[i] * present_velocity_values[q]) * permeability * 2.0)) *
                                    ad_ns_fe_values.JxW(q);
                                break;
                            }

                            case PRM::inlet_pressure_constraint:
                                break;

                            case PRM::pump_power_constraint:
                                break;

                            case PRM::only_volume_constraint:
                                break;

                            default:
                                break;
                            }

                            break;
                        }

                        case PRM::lagrange_field_cstr_function:
                        {
                            switch (field_constraint_function)
                            {
                            case PRM::no_field_constraints:
                                break;

                            case PRM::norm_xphys_heaviside_gradients_constraint:
                                break;

                            case PRM::abs_cos_u_grad_xphys_constraint:
                            {
                                if (std::abs(present_velocity_values[q] * xphys_gradients[q]) > field_if_epsilon)
                                {
                                    local_rhs(i) -=
                                        (-1.0) *
                                        ((phi_u[i]) *
                                         ((((present_velocity_values[q] * xphys_gradients[q]) /
                                            (std::abs(present_velocity_values[q] * xphys_gradients[q]) + field_plus_epsilon) *
                                            present_velocity_values[q].norm() *
                                            xphys_gradients[q].norm()) *
                                           xphys_gradients[q]) -
                                          ((std::abs(present_velocity_values[q] * xphys_gradients[q]) *
                                            xphys_gradients[q].norm() / present_velocity_values[q].norm()) *
                                           present_velocity_values[q])) /
                                         (present_velocity_values[q].norm_square() * xphys_gradients[q].norm_square() + field_plus_epsilon) /
                                         volume_total) *
                                        ad_ns_fe_values.JxW(q);
                                }
                                break;
                            }

                            case PRM::xphys_abs_cos_u_grad_xphys_constraint:
                            {
                                if (std::abs(present_velocity_values[q] * xphys_gradients[q]) > field_if_epsilon)
                                {
                                    local_rhs(i) -=
                                        (-1.0) *
                                        ((phi_u[i]) *
                                         ((((present_velocity_values[q] * xphys_gradients[q]) /
                                            (std::abs(present_velocity_values[q] * xphys_gradients[q]) + field_plus_epsilon) *
                                            present_velocity_values[q].norm() *
                                            xphys_gradients[q].norm()) *
                                           xphys_gradients[q]) -
                                          ((std::abs(present_velocity_values[q] * xphys_gradients[q]) *
                                            xphys_gradients[q].norm() / present_velocity_values[q].norm()) *
                                           present_velocity_values[q])) /
                                         (present_velocity_values[q].norm_square() * xphys_gradients[q].norm_square() + field_plus_epsilon) *
                                         xphys_values[q] /
                                         volume_total) *
                                        ad_ns_fe_values.JxW(q);
                                }
                                break;
                            }

                            case PRM::norm_u_grad_xphys_constraint:
                            {
                                if (std::abs(present_velocity_values[q] * xphys_gradients[q]) > field_if_epsilon)
                                {
                                    local_rhs(i) -=
                                        (-1.0) *
                                        ((phi_u[i]) *
                                         ((((present_velocity_values[q] * xphys_gradients[q]) /
                                            (std::abs(present_velocity_values[q] * xphys_gradients[q]) + field_plus_epsilon)) *
                                           xphys_gradients[q])) /
                                         volume_total) *
                                        ad_ns_fe_values.JxW(q);
                                }
                                break;
                            }

                            case PRM::fnorm_xphys_filter_hessians_constraint:
                            {
                                break;
                            }

                            case PRM::norm_xphys_filter_gradients_constraint:
                            {
                                break;
                            }

                            default:
                                break;
                            }
                            break;
                        }

                        default:
                            break;
                        }
                    }
                }

                for (const auto face : cell->face_indices())
                {
                    auto face_iter = cell->face(face);
                    switch (lag_function)
                    {
                    case PRM::lagrange_flow_cstr_function:
                    { /* code */
                        switch (flow_constraint_function)
                        {
                        case PRM::flow_dissipation_constraint:
                            break;

                        case PRM::inlet_pressure_constraint:
                        {
                            if (face_iter->at_boundary() && face_iter->boundary_id() == 1)
                            {
                                fe_face_values.reinit(cell, face);
                                for (unsigned int q = 0; q < n_face_q_points; ++q)
                                {
                                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                    {
                                        local_rhs(i) -=
                                            (fe_face_values[pressure].value(i, q)) * 1.0 *
                                            fe_face_values.JxW(q);
                                    }
                                }
                            }
                            break;
                        }

                        case PRM::pump_power_constraint:
                        {
                            if (face_iter->at_boundary() && (face_iter->boundary_id() == 1 || face_iter->boundary_id() == 2))
                            {
                                fe_face_values.reinit(cell, face);
                                ns_fe_face_values.reinit(ns_cell, face);

                                ns_fe_face_values[pressure].get_function_values(ns_solution_relevant,
                                                                                present_face_pressure_values);
                                ns_fe_face_values[velocities].get_function_values(ns_solution_relevant,
                                                                                  present_face_velocity_values);
                                present_face_normal_vectors = fe_face_values.get_normal_vectors();

                                for (unsigned int q = 0; q < n_face_q_points; ++q)
                                {
                                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                    {
                                        local_rhs(i) -=
                                            ((fe_face_values[velocities].value(i, q) *
                                              (((-1.0) *
                                                (present_face_pressure_values[q] + 0.5 * present_face_velocity_values[q].norm_square()) *
                                                present_face_normal_vectors[q]) +
                                               ((-1.0) *
                                                present_face_velocity_values[q] *
                                                (present_face_velocity_values[q] * present_face_normal_vectors[q])))) +
                                             (fe_face_values[pressure].value(i, q) *
                                              ((-1.0) *
                                               (present_face_velocity_values[q] * present_face_normal_vectors[q])))) *
                                            fe_face_values.JxW(q);
                                    }
                                }
                            }
                            break;
                        }

                        case PRM::only_volume_constraint:
                            break;

                        default:
                            break;
                        }

                        break;
                    }

                    default:
                        break;
                    }
                }

                cell->get_dof_indices(local_dof_indices);

                if (assemble_matrix)
                {
                    ad_ns_constraints.distribute_local_to_global(local_matrix,
                                                                 local_rhs,
                                                                 local_dof_indices,
                                                                 ad_ns_system_matrix,
                                                                 ad_ns_system_rhs);
                }
                else
                {
                    ad_ns_constraints.distribute_local_to_global(local_rhs,
                                                                 local_dof_indices,
                                                                 ad_ns_system_rhs);
                }
            }
        }
        if (assemble_matrix)
        {
            ad_ns_system_matrix.compress(VectorOperation::add);
        }

        ad_ns_system_rhs.compress(VectorOperation::add);
    }

    void TopOptHeatFlow::ad_continuous_ns_assemble_with_residual_linearization(const PRM::LagrangeFunctionType lag_function, const bool assemble_matrix)
    {
        TimerOutput::Scope t(computing_timer, "ad_ns_assembly_with_residual_linearization");

        if (assemble_matrix)
        {
            ad_ns_system_matrix = 0;
        }

        ad_ns_system_rhs = 0;

        using ADHelper = Differentiation::AD::ResidualLinearization<Differentiation::AD::NumberTypes::sacado_dfad, double>;
        using ADNumberType = typename ADHelper::ad_type;

        QGauss<dim> quadrature_formula(quadrature_degree + 1);
        QGauss<dim - 1> face_quadrature_formula(quadrature_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values | update_gradients);
        FEValues<dim> temp_fe_values(temp_fe,
                                     quadrature_formula,
                                     update_values | update_gradients);
        FEValues<dim> ad_temp_fe_values(ad_temp_fe,
                                        quadrature_formula,
                                        update_values | update_gradients);
        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values | update_gradients);
        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);
        FEValues<dim> ad_ns_fe_values(ad_ns_fe,
                                      quadrature_formula,
                                      update_values | update_quadrature_points |
                                          update_JxW_values | update_gradients);

        FEFaceValues<dim> ns_fe_face_values(ns_fe,
                                            face_quadrature_formula,
                                            update_values);
        FEFaceValues<dim> ad_ns_fe_face_values(ad_ns_fe,
                                               face_quadrature_formula,
                                               update_values | update_normal_vectors |
                                                   update_JxW_values);

        const unsigned int dofs_per_cell = ad_ns_fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        const FEValuesExtractors::Vector ad_velocities(0);
        const FEValuesExtractors::Scalar ad_pressure(dim);

        const unsigned int n_independent_variables = dofs_per_cell;
        const unsigned int n_dependent_variables = dofs_per_cell + 1;

        ADHelper ad_helper(n_independent_variables, n_dependent_variables);

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        FullMatrix<double> jacobian_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*温度方程右端项-热源项*/
        TempRightHandSide<dim> temp_rhs;

        /*预分配内存*/
        std::vector<Tensor<1, dim, ADNumberType>> present_velocity_values(n_q_points);
        std::vector<Tensor<2, dim, ADNumberType>> present_velocity_gradients(n_q_points);
        std::vector<SymmetricTensor<2, dim, ADNumberType>> present_velocity_symmetric_gradients(n_q_points);
        std::vector<ADNumberType> present_pressure_values(n_q_points);
        std::vector<Tensor<1, dim, ADNumberType>> present_pressure_gradients(n_q_points);
        std::vector<Tensor<1, dim>> present_temperature_gradients(n_q_points);
        std::vector<double> present_temperature_values(n_q_points);

        std::vector<double> present_ad_temp_values(n_q_points);
        std::vector<Tensor<1, dim>> present_ad_temp_gradients(n_q_points);
        std::vector<double> xphys_values(n_q_points);
        std::vector<Tensor<1, dim>> xphys_gradients(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);
        std::vector<double> temp_rhs_values(n_q_points);

        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);

        std::vector<double> phi_p(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

        /*SUPG-PSPG-LSIC稳定化参数及其对速度矢量的偏导*/
        std::vector<ADNumberType> tauSUPG(n_q_points), tauPSPG(n_q_points), vLSIC(n_q_points), tauSUPGT(n_q_points);

        StabilizationParameter<dim, ADNumberType> stabilization_parameter;
        double kinematic_viscosity = viscosity / rho;
        const double thermal_diffusivity = thermal_cond_fluid / (rho * capacity);

        /*迭代器*/
        auto cell = ad_ns_dof_handler.begin_active();
        const auto endc = ad_ns_dof_handler.end();
        auto ns_cell = ns_dof_handler.begin_active();
        auto temp_cell = temp_dof_handler.begin_active();
        auto ad_temp_cell = ad_temp_dof_handler.begin_active();
        auto top_cell = top_dof_handler.begin_active();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++ns_cell, ++temp_cell, ++ad_temp_cell, ++top_cell, ++top_equivalent_cell)
        {
            if (cell->is_locally_owned())
            {

                ad_ns_fe_values.reinit(cell);
                ns_fe_values.reinit(ns_cell);
                temp_fe_values.reinit(temp_cell);
                ad_temp_fe_values.reinit(ad_temp_cell);
                top_fe_values.reinit(top_cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                local_matrix = 0;
                jacobian_matrix = 0;
                local_rhs = 0;

                cell->get_dof_indices(local_dof_indices);

                ad_helper.reset(n_independent_variables, n_dependent_variables);
                ad_helper.register_dof_values(ns_solution_relevant, local_dof_indices);

                const std::vector<ADNumberType> &ns_dof_values_ad = ad_helper.get_sensitive_dof_values();

                /*读取高斯点处的函数值*/
                ns_fe_values[velocities].get_function_values_from_local_dof_values(ns_dof_values_ad,
                                                                                   present_velocity_values);
                ns_fe_values[velocities].get_function_gradients_from_local_dof_values(ns_dof_values_ad,
                                                                                      present_velocity_gradients);

                ns_fe_values[velocities].get_function_symmetric_gradients_from_local_dof_values(
                    ns_dof_values_ad, present_velocity_symmetric_gradients);

                ns_fe_values[pressure].get_function_values_from_local_dof_values(ns_dof_values_ad,
                                                                                 present_pressure_values);
                ns_fe_values[pressure].get_function_gradients_from_local_dof_values(ns_dof_values_ad,
                                                                                    present_pressure_gradients);
                temp_fe_values.get_function_gradients(temp_solution_relevant,
                                                      present_temperature_gradients);
                temp_fe_values.get_function_values(temp_solution_relevant,
                                                   present_temperature_values);
                ad_temp_fe_values.get_function_values(ad_temp_solution_relevant,
                                                      present_ad_temp_values);
                ad_temp_fe_values.get_function_gradients(ad_temp_solution_relevant,
                                                         present_ad_temp_gradients);
                auto &present_top_fe_values = top_fe_values.get_present_fe_values();
                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_fe_values.get_function_values(xphys_heaviside_relevant,
                                                          xphys_values);
                present_top_fe_values.get_function_gradients(xphys_heaviside_relevant,
                                                             xphys_gradients);
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);

                double element_size = (cell->diameter()) / std::sqrt(2);

                stabilization_parameter.value(present_velocity_values,
                                              kinematic_viscosity, element_size,
                                              tauSUPG, tauPSPG, vLSIC);
                stabilization_parameter.value_temperature(present_velocity_values,
                                                          thermal_diffusivity, element_size, tauSUPGT);

                std::vector<ADNumberType> residual_ad(n_dependent_variables, ADNumberType(0.0));

                /*SUPG-PSPG-LSIC修正  局部矩阵组装*/
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);
                    temp_rhs_values[q] = temp_rhs.value(ad_ns_fe_values.quadrature_point(q));
                    permeability = material_interpolate.permeability_interpolate(xphys_equivalent_values_q_point);
                    thermal_conductivity = material_interpolate.conductivity_interpolate(xphys_equivalent_values_q_point);
                    heatsource_coeff = material_interpolate.heatsource_interpolate(xphys_equivalent_values_q_point);
                    convection_coeff = material_interpolate.convection_interpolate(xphys_equivalent_values_q_point);
                    ADNumberType present_velocity_divergence = trace(present_velocity_gradients[q]);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        div_phi_u[k] = ad_ns_fe_values[ad_velocities].divergence(k, q);
                        grad_phi_u[k] = ad_ns_fe_values[ad_velocities].gradient(k, q);

                        phi_u[k] = ad_ns_fe_values[ad_velocities].value(k, q);
                        phi_p[k] = ad_ns_fe_values[ad_pressure].value(k, q);
                        grad_phi_p[k] = ad_ns_fe_values[ad_pressure].gradient(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        if (assemble_matrix)
                        {
                            residual_ad[i] +=
                                ((phi_u[i]) *
                                     ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                      (permeability * present_velocity_values[q])) +
                                 (2 * viscosity * scalar_product(grad_phi_u[i], present_velocity_symmetric_gradients[q])) -
                                 (div_phi_u[i] * present_pressure_values[q]) +
                                 (phi_p[i] * present_velocity_divergence) +
                                 ((tauSUPG[q] * (grad_phi_u[i] * present_velocity_values[q])) + (grad_phi_p[i] * (1.0 / rho) * tauPSPG[q])) *
                                     ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                      present_pressure_gradients[q] +
                                      (permeability * present_velocity_values[q])) +
                                 (vLSIC[q] * rho * div_phi_u[i] * present_velocity_divergence)) *
                                ad_ns_fe_values.JxW(q);
                        }
                    }

                    switch (lag_function)
                    {
                    case PRM::lagrange_obj_function:
                    {
                        residual_ad[n_dependent_variables - 1] -=
                            ((present_ad_temp_values[q] +
                              tauSUPGT[q] * (present_velocity_values[q] * present_ad_temp_gradients[q])) *
                                 ((convection_coeff * rho * capacity * (present_velocity_values[q] * present_temperature_gradients[q])) -
                                  (heatsource_coeff * (t_Q - present_temperature_values[q]) + temp_rhs_values[q])) +
                             present_ad_temp_gradients[q] * present_temperature_gradients[q] * thermal_conductivity) *
                            ad_ns_fe_values.JxW(q);

                        switch (objective_function)
                        {
                        case PRM::mean_temp_objective:
                            break;

                        case PRM::abs_u_gradt_objective:
                        {
                        }

                        case PRM::norm_gradt_objective:
                            break;

                        case PRM::mean_solid_temp_objective:
                            break;

                        case PRM::thermal_compliance_objective:
                        {
                        }

                        case PRM::heat_exchange_objective:
                            break;

                        case PRM::mean_boundary_temp_objective:
                            break;

                        case PRM::xphys_abs_cos_u_grad_temp_objective:
                        {
                            if (std::abs(present_velocity_values[q] * present_temperature_gradients[q]) > field_if_epsilon)
                            {
                            }
                            break;
                        }

                        default:
                            break;
                        }

                        break;
                    }

                    case PRM::lagrange_flow_cstr_function:
                    {
                        switch (flow_constraint_function)
                        {
                        case PRM::flow_dissipation_constraint:
                        {
                            residual_ad[n_dependent_variables - 1] -=
                                ((0.5 * viscosity * scalar_product(present_velocity_gradients[q], present_velocity_gradients[q])) +
                                 ((present_velocity_values[q] * present_velocity_values[q]) * permeability)) *
                                ad_ns_fe_values.JxW(q);
                            break;
                        }

                        case PRM::inlet_pressure_constraint:
                            break;

                        case PRM::pump_power_constraint:
                            break;

                        case PRM::only_volume_constraint:
                            break;

                        default:
                            break;
                        }

                        break;
                    }

                    case PRM::lagrange_field_cstr_function:
                    {
                        switch (field_constraint_function)
                        {
                        case PRM::no_field_constraints:
                            break;

                        case PRM::norm_xphys_heaviside_gradients_constraint:
                            break;

                        case PRM::abs_cos_u_grad_xphys_constraint:
                        {

                            break;
                        }

                        case PRM::xphys_abs_cos_u_grad_xphys_constraint:
                        {

                            break;
                        }

                        case PRM::norm_u_grad_xphys_constraint:
                        {

                            break;
                        }

                        case PRM::fnorm_xphys_filter_hessians_constraint:
                        {
                            break;
                        }

                        case PRM::norm_xphys_filter_gradients_constraint:
                        {
                            break;
                        }

                        default:
                            break;
                        }
                        break;
                    }

                    default:
                        break;
                    }
                }

                for (const auto face : cell->face_indices())
                {
                    auto face_iter = cell->face(face);
                    switch (lag_function)
                    {
                    case PRM::lagrange_obj_function:
                    {
                        if (face_iter->at_boundary() && face_iter->boundary_id() == 4)
                        {
                            ad_ns_fe_face_values.reinit(cell, face);
                            for (unsigned int q = 0; q < n_face_q_points; ++q)
                            {
                                residual_ad[n_dependent_variables - 1] -=
                                    (-1.0) *
                                    (present_ad_temp_values[q] * heatflux0) *
                                    ad_ns_fe_face_values.JxW(q);
                            }
                        }
                        break;
                    }
                    case PRM::lagrange_flow_cstr_function:
                    { /* code */
                        switch (flow_constraint_function)
                        {
                        case PRM::flow_dissipation_constraint:
                            break;

                        case PRM::inlet_pressure_constraint:
                        {

                            break;
                        }

                        case PRM::pump_power_constraint:
                        {

                            break;
                        }

                        case PRM::only_volume_constraint:
                            break;

                        default:
                            break;
                        }

                        break;
                    }

                    default:
                        break;
                    }
                }

                ad_helper.register_residual_vector(residual_ad);
                ad_helper.compute_linearization(jacobian_matrix);
                if (assemble_matrix)
                {
                    for (unsigned int i = 0; i < dofs_per_cell; i++)
                        for (unsigned int j = 0; j < dofs_per_cell; j++)
                        {
                            local_matrix[i][j] = jacobian_matrix[j][i];
                        }
                }
                for (unsigned int j = 0; j < dofs_per_cell; j++)
                {
                    local_rhs[j] = jacobian_matrix[n_dependent_variables - 1][j];
                }

                if (assemble_matrix)
                {
                    ad_ns_constraints.distribute_local_to_global(local_matrix,
                                                                 local_rhs,
                                                                 local_dof_indices,
                                                                 ad_ns_system_matrix,
                                                                 ad_ns_system_rhs);
                }
                else
                {
                    ad_ns_constraints.distribute_local_to_global(local_rhs,
                                                                 local_dof_indices,
                                                                 ad_ns_system_rhs);
                }
            }
        }
        if (assemble_matrix)
        {
            ad_ns_system_matrix.compress(VectorOperation::add);
        }

        ad_ns_system_rhs.compress(VectorOperation::add);
    }

    /*连续伴随NS求解*/
    void TopOptHeatFlow::ad_continuous_ns_solve(const PRM::LagrangeFunctionType lag_function)
    {
        TimerOutput::Scope t(computing_timer, "ad_ns_solve");
        /* PETSc Preconditioner*/

        SolverControl solver_control(ad_ns_system_matrix.m(),
                                     1e-8 * ad_ns_system_rhs.l2_norm(),
                                     true);

        PETScWrappers::SparseDirectMUMPS solver(solver_control);

        LA::MPI::Vector &distributed_solution =
            (lag_function == PRM::lagrange_obj_function)          ? ad_ns_solution_distributed
            : (lag_function == PRM::lagrange_flow_cstr_function)  ? ad_ns_flow_solution_distributed
            : (lag_function == PRM::lagrange_field_cstr_function) ? ad_ns_field_solution_distributed

                                                                  : ad_ns_solution_distributed;

        pcout << std::endl
              << "******************************************************************************" << std::endl;
        pcout << "Solving For Adjoint Continuous Navier-Stokes Field... " << std::endl;

        solver.solve(ad_ns_system_matrix, distributed_solution, ad_ns_system_rhs);

        pcout << "Solver Converged after steps: " << solver_control.last_step() << std::endl;

        ad_ns_constraints.distribute(distributed_solution);

        switch (lag_function)
        {
        case PRM::lagrange_obj_function:
        {
            ad_ns_solution_relevant = distributed_solution;
            break;
        }
        case PRM::lagrange_flow_cstr_function:
        {
            ad_ns_flow_solution_relevant = distributed_solution;
            break;
        }
        case PRM::lagrange_field_cstr_function:
        {
            ad_ns_field_solution_relevant = distributed_solution;
            break;
        }
        default:
            break;
        }
    }

}