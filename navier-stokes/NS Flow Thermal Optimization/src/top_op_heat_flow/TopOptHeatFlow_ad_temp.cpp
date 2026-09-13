#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*组装温度连续伴随*/
    void TopOptHeatFlow::ad_continuous_temp_assemble()
    {
        TimerOutput::Scope t(computing_timer, "ad_temp_assembly");

        ad_temp_system_matrix = 0;
        ad_temp_system_rhs = 0;

        QGauss<dim> quadrature_formula(quadrature_degree + 1);
        QGauss<dim - 1> face_quadrature_formula(quadrature_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values);
        FEValues<dim> temp_fe_values(temp_fe,
                                     quadrature_formula,
                                     update_gradients);
        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);
        FEValues<dim> ad_temp_fe_values(ad_temp_fe,
                                        quadrature_formula,
                                        update_values | update_quadrature_points | update_hessians |
                                            update_JxW_values | update_gradients);
        FEFaceValues<dim> ad_temp_fe_face_values(ad_temp_fe,
                                                 face_quadrature_formula,
                                                 update_values | update_JxW_values);

        const unsigned int dofs_per_cell = ad_temp_fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*伴随温度方程右端项*/
        double volume_total = GridTools::volume(triangulation);
        double area_heatflux = boundary_area[4];

        /*预分配内存*/
        std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
        std::vector<Tensor<1, dim>> present_temperature_gradients(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);

        std::vector<double> phi_t(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_t(dofs_per_cell);

        /*SUPG-PSPG-LSIC稳定化参数*/
        std::vector<double> tauSUPGT(n_q_points);
        StabilizationParameter<dim> stabilization_parameter;
        const double thermal_diffusivity = thermal_cond_fluid / (rho * capacity);

        /*迭代器*/
        auto cell = ad_temp_dof_handler.begin_active();
        const auto endc = ad_temp_dof_handler.end();
        auto ns_cell = ns_dof_handler.begin_active();
        auto temp_cell = temp_dof_handler.begin_active();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++ns_cell, ++temp_cell, ++top_equivalent_cell)
        {
            if (cell->is_locally_owned())
            {
                ad_temp_fe_values.reinit(cell);
                ns_fe_values.reinit(ns_cell);
                temp_fe_values.reinit(temp_cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                local_matrix = 0;
                local_rhs = 0;

                ns_fe_values[velocities].get_function_values(ns_solution_relevant,
                                                             present_velocity_values);
                temp_fe_values.get_function_gradients(temp_solution_relevant,
                                                      present_temperature_gradients);

                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);

                double element_size = (cell->diameter()) / std::sqrt(2);

                stabilization_parameter.value_temperature(present_velocity_values,
                                                          thermal_diffusivity, element_size, tauSUPGT);

                /*SUPG-PSPG-LSIC修正  局部矩阵组装*/
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);
                    thermal_conductivity = material_interpolate.conductivity_interpolate(xphys_equivalent_values_q_point);
                    heatsource_coeff = material_interpolate.heatsource_interpolate(xphys_equivalent_values_q_point);
                    convection_coeff = material_interpolate.convection_interpolate(xphys_equivalent_values_q_point);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        phi_t[k] = ad_temp_fe_values.shape_value(k, q);
                        grad_phi_t[k] = ad_temp_fe_values.shape_grad(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            local_matrix(i, j) +=
                                (phi_t[j] * (convection_coeff * rho * capacity * (present_velocity_values[q] * grad_phi_t[i]) +
                                             heatsource_coeff * phi_t[i]) +
                                 grad_phi_t[j] * grad_phi_t[i] * thermal_conductivity +
                                 (tauSUPGT[q] * (present_velocity_values[q] * grad_phi_t[j])) *
                                     (convection_coeff * rho * capacity * (present_velocity_values[q] * grad_phi_t[i]) +
                                      heatsource_coeff * phi_t[i])) *
                                ad_temp_fe_values.JxW(q);
                        }

                        switch (objective_function)
                        {
                        case PRM::mean_temp_objective:
                        {
                            local_rhs(i) -=
                                (phi_t[i] * 1.0 / volume_total) *
                                ad_temp_fe_values.JxW(q);
                            break;
                        }

                        case PRM::abs_u_gradt_objective:
                        {
                            local_rhs(i) -=
                                (-1.0) *
                                ((grad_phi_t[i] * present_velocity_values[q]) *
                                 (present_velocity_values[q] * present_temperature_gradients[q]) /
                                 std::abs(present_velocity_values[q] * present_temperature_gradients[q])) *
                                ad_temp_fe_values.JxW(q);
                            break;
                        }

                        case PRM::norm_gradt_objective:
                        {
                            local_rhs(i) -=
                                ((grad_phi_t[i] * present_temperature_gradients[q]) / present_temperature_gradients[q].norm()) *
                                ad_temp_fe_values.JxW(q);
                            break;
                        }

                        case PRM::mean_solid_temp_objective:
                        {
                            local_rhs(i) -=
                                (phi_t[i] * (1 - xphys_equivalent_values_q_point) * 1.0 / volume_total) *
                                ad_temp_fe_values.JxW(q);
                            break;
                        }

                        case PRM::thermal_compliance_objective:
                        {
                            local_rhs(i) -=
                                (((grad_phi_t[i] * present_temperature_gradients[q]) * thermal_conductivity * 2.0) +
                                 (phi_t[i] * rho * capacity * (present_velocity_values[q] * present_temperature_gradients[q]))) *
                                ad_temp_fe_values.JxW(q);
                            break;
                        }

                        case PRM::heat_exchange_objective:
                        {
                            local_rhs(i) -=
                                (-1.0) *
                                (phi_t[i] * (-1.0) * heatsource_coeff) *
                                ad_temp_fe_values.JxW(q);
                            break;
                        }

                        case PRM::mean_boundary_temp_objective:
                            break;

                        case PRM::xphys_abs_cos_u_grad_temp_objective:
                        {
                            if (std::abs(present_velocity_values[q] * present_temperature_gradients[q]) > field_if_epsilon)
                            {
                                local_rhs(i) -=
                                    (-1.0) *
                                    ((grad_phi_t[i]) *
                                     ((((present_velocity_values[q] * present_temperature_gradients[q]) /
                                        (std::abs(present_velocity_values[q] * present_temperature_gradients[q]) + field_plus_epsilon) *
                                        present_velocity_values[q].norm() * present_temperature_gradients[q].norm()) *
                                       present_velocity_values[q]) -
                                      ((std::abs(present_velocity_values[q] * present_temperature_gradients[q]) *
                                        present_velocity_values[q].norm() / (present_temperature_gradients[q].norm() + field_plus_epsilon)) *
                                       present_temperature_gradients[q])) /
                                     (present_velocity_values[q].norm_square() * present_temperature_gradients[q].norm_square() + field_plus_epsilon) *
                                     xphys_equivalent_values[q] /
                                     volume_total) *
                                    ad_temp_fe_values.JxW(q);
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

                    switch (objective_function)
                    {
                    case PRM::mean_temp_objective:
                        break;

                    case PRM::abs_u_gradt_objective:
                        break;

                    case PRM::norm_gradt_objective:
                        break;
                    case PRM::mean_solid_temp_objective:
                        break;
                    case PRM::thermal_compliance_objective:
                        break;

                    case PRM::heat_exchange_objective:
                        break;

                    case PRM::mean_boundary_temp_objective:
                    {
                        if (face_iter->at_boundary() && face_iter->boundary_id() == 4)
                        {
                            ad_temp_fe_face_values.reinit(cell, face);
                            for (unsigned int q = 0; q < n_face_q_points; ++q)
                            {
                                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                                {
                                    local_rhs(i) -=
                                        (ad_temp_fe_face_values.shape_value(i, q)) * (1.0 / area_heatflux) *
                                        ad_temp_fe_face_values.JxW(q);
                                }
                            }
                        }
                        break;
                    }

                    case PRM::xphys_abs_cos_u_grad_temp_objective:
                        break;

                    default:
                        break;
                    }
                }

                cell->get_dof_indices(local_dof_indices);

                ad_temp_constraints.distribute_local_to_global(local_matrix,
                                                               local_rhs,
                                                               local_dof_indices,
                                                               ad_temp_system_matrix,
                                                               ad_temp_system_rhs);
            }
        }

        ad_temp_system_rhs.compress(VectorOperation::add);

        ad_temp_system_matrix.compress(VectorOperation::add);
    }

    void TopOptHeatFlow::ad_continuous_temp_assemble_with_residual_linearization()
    {
        TimerOutput::Scope t(computing_timer, "ad_temp_assembly_with_residual_linearization");

        ad_temp_system_matrix = 0;
        ad_temp_system_rhs = 0;

        using ADHelper = Differentiation::AD::ResidualLinearization<Differentiation::AD::NumberTypes::sacado_dfad, double>;
        using ADNumberType = typename ADHelper::ad_type;

        QGauss<dim> quadrature_formula(quadrature_degree + 1);
        QGauss<dim - 1> face_quadrature_formula(quadrature_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values);
        FEValues<dim> temp_fe_values(temp_fe,
                                     quadrature_formula,
                                     update_values | update_gradients);
        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);
        FEValues<dim> ad_temp_fe_values(ad_temp_fe,
                                        quadrature_formula,
                                        update_values | update_quadrature_points | update_hessians |
                                            update_JxW_values | update_gradients);
        FEFaceValues<dim> ad_temp_fe_face_values(ad_temp_fe,
                                                 face_quadrature_formula,
                                                 update_values | update_JxW_values);

        const unsigned int dofs_per_cell = ad_temp_fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);
        const FEValuesExtractors::Scalar temperature(0);

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
        std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
        std::vector<ADNumberType> present_temperature_values(n_q_points);
        std::vector<Tensor<1, dim, ADNumberType>> present_temperature_gradients(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);
        std::vector<double> temp_rhs_values(n_q_points);

        std::vector<double> phi_t(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_t(dofs_per_cell);

        /*SUPG-PSPG-LSIC稳定化参数*/
        std::vector<double> tauSUPGT(n_q_points);
        StabilizationParameter<dim> stabilization_parameter;
        const double thermal_diffusivity = thermal_cond_fluid / (rho * capacity);

        /*迭代器*/
        auto cell = ad_temp_dof_handler.begin_active();
        const auto endc = ad_temp_dof_handler.end();
        auto ns_cell = ns_dof_handler.begin_active();
        auto temp_cell = temp_dof_handler.begin_active();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++ns_cell, ++temp_cell, ++top_equivalent_cell)
        {
            if (cell->is_locally_owned())
            {

                ad_temp_fe_values.reinit(cell);
                ns_fe_values.reinit(ns_cell);
                temp_fe_values.reinit(temp_cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                local_matrix = 0;
                jacobian_matrix = 0;
                local_rhs = 0;

                cell->get_dof_indices(local_dof_indices);

                ad_helper.reset(n_independent_variables, n_dependent_variables);
                ad_helper.register_dof_values(temp_solution_relevant, local_dof_indices);
                const std::vector<ADNumberType> &temp_dof_values_ad = ad_helper.get_sensitive_dof_values();

                ns_fe_values[velocities].get_function_values(ns_solution_relevant,
                                                             present_velocity_values);

                temp_fe_values[temperature].get_function_values_from_local_dof_values(temp_dof_values_ad,
                                                                                      present_temperature_values);
                temp_fe_values[temperature].get_function_gradients_from_local_dof_values(temp_dof_values_ad,
                                                                                         present_temperature_gradients);

                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);

                double element_size = (cell->diameter()) / std::sqrt(2);

                stabilization_parameter.value_temperature(present_velocity_values,
                                                          thermal_diffusivity, element_size, tauSUPGT);
                std::vector<ADNumberType> residual_ad(n_dependent_variables, ADNumberType(0.0));
                /*SUPG-PSPG-LSIC修正  局部矩阵组装*/
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);
                    temp_rhs_values[q] = temp_rhs.value(ad_temp_fe_values.quadrature_point(q));
                    thermal_conductivity = material_interpolate.conductivity_interpolate(xphys_equivalent_values_q_point);
                    heatsource_coeff = material_interpolate.heatsource_interpolate(xphys_equivalent_values_q_point);
                    convection_coeff = material_interpolate.convection_interpolate(xphys_equivalent_values_q_point);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        phi_t[k] = ad_temp_fe_values.shape_value(k, q);
                        grad_phi_t[k] = ad_temp_fe_values.shape_grad(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {

                        residual_ad[i] +=
                            ((phi_t[i] + tauSUPGT[q] * (present_velocity_values[q] * grad_phi_t[i])) *
                                 ((convection_coeff * rho * capacity * (present_velocity_values[q] * present_temperature_gradients[q])) -
                                  (heatsource_coeff * (t_Q - present_temperature_values[q]) + temp_rhs_values[q])) +
                             (grad_phi_t[i] * present_temperature_gradients[q] * thermal_conductivity)) *
                            ad_temp_fe_values.JxW(q);
                    }

                    switch (objective_function)
                    {
                    case PRM::mean_temp_objective:
                    {

                        break;
                    }

                    case PRM::abs_u_gradt_objective:
                    {

                        break;
                    }

                    case PRM::norm_gradt_objective:
                    {

                        break;
                    }

                    case PRM::mean_solid_temp_objective:
                    {

                        break;
                    }

                    case PRM::thermal_compliance_objective:
                    {

                        break;
                    }

                    case PRM::heat_exchange_objective:
                    {
                        residual_ad[n_dependent_variables - 1] -=
                            (-1.0) *
                            heatsource_coeff *
                            (t_Q - present_temperature_values[q]) *
                            ad_temp_fe_values.JxW(q);
                        break;
                    }

                    case PRM::mean_boundary_temp_objective:
                        break;

                    case PRM::xphys_abs_cos_u_grad_temp_objective:
                    {

                        break;
                    }

                    default:
                        break;
                    }
                }

                for (const auto face : cell->face_indices())
                {
                    auto face_iter = cell->face(face);
                    if (face_iter->at_boundary() && face_iter->boundary_id() == 4)
                    {
                        ad_temp_fe_face_values.reinit(cell, face);
                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                                residual_ad[i] -=
                                    (-1.0) *
                                    ((ad_temp_fe_face_values.shape_value(i, q)) * heatflux0) *
                                    ad_temp_fe_face_values.JxW(q);
                            }
                        }
                    }

                    switch (objective_function)
                    {
                    case PRM::mean_temp_objective:
                        break;

                    case PRM::abs_u_gradt_objective:
                        break;

                    case PRM::norm_gradt_objective:
                        break;
                    case PRM::mean_solid_temp_objective:
                        break;
                    case PRM::thermal_compliance_objective:
                        break;

                    case PRM::heat_exchange_objective:
                        break;

                    case PRM::mean_boundary_temp_objective:
                    {

                        break;
                    }

                    case PRM::xphys_abs_cos_u_grad_temp_objective:
                        break;

                    default:
                        break;
                    }
                }

                ad_helper.register_residual_vector(residual_ad);
                ad_helper.compute_linearization(jacobian_matrix);

                for (unsigned int i = 0; i < dofs_per_cell; i++)
                    for (unsigned int j = 0; j < dofs_per_cell; j++)
                    {
                        local_matrix[i][j] = jacobian_matrix[j][i];
                    }

                for (unsigned int j = 0; j < dofs_per_cell; j++)
                {
                    local_rhs[j] = jacobian_matrix[n_dependent_variables - 1][j];
                }

                ad_temp_constraints.distribute_local_to_global(local_matrix,
                                                               local_rhs,
                                                               local_dof_indices,
                                                               ad_temp_system_matrix,
                                                               ad_temp_system_rhs);
            }
        }

        ad_temp_system_rhs.compress(VectorOperation::add);

        ad_temp_system_matrix.compress(VectorOperation::add);
    }

    /*连续伴随温度求解*/
    void TopOptHeatFlow::ad_continuous_temp_solve()
    {
        TimerOutput::Scope t(computing_timer, "ad_temp_solve");
        /* PETSc Preconditioner*/
        using PreconditionType = PETScWrappers::PreconditionBlockJacobi;
        PreconditionType preconditioner;
        {
            PreconditionType::AdditionalData data;
            preconditioner.initialize(ad_temp_system_matrix, data);
        }

        SolverControl solver_control(ad_temp_system_matrix.m(),
                                     1e-8 * ad_temp_system_rhs.l2_norm(),
                                     true);

        PETScWrappers::SparseDirectMUMPS solver(solver_control);

        pcout << std::endl
              << "******************************************************************************" << std::endl;
        pcout << "Solving For Adjoint Continuous Temperature Field... " << std::endl;

        solver.solve(ad_temp_system_matrix, ad_temp_solution_distributed, ad_temp_system_rhs);

        pcout << "Solver Converged after steps: " << solver_control.last_step() << std::endl;

        ad_temp_constraints.distribute(ad_temp_solution_distributed);
        ad_temp_solution_relevant = ad_temp_solution_distributed;
    }

}