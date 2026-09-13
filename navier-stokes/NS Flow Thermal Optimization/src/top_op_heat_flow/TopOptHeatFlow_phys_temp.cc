#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*组装对流扩散方程*/
    void TopOptHeatFlow::temp_assemble()
    {
        TimerOutput::Scope t(computing_timer, "temp_assembly");

        temp_system_matrix = 0;
        temp_system_rhs = 0;

        QGauss<dim> quadrature_formula(quadrature_degree + 1);
        QGauss<dim - 1> face_quadrature_formula(quadrature_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values);
        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);
        FEValues<dim> temp_fe_values(temp_fe,
                                     quadrature_formula,
                                     update_values | update_quadrature_points | update_hessians |
                                         update_JxW_values | update_gradients);
        FEFaceValues<dim> temp_fe_face_values(temp_fe,
                                              face_quadrature_formula,
                                              update_values | update_JxW_values);

        const unsigned int dofs_per_cell = temp_fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();
        const unsigned int n_face_q_points = face_quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*温度方程右端项-热源项*/
        TempRightHandSide<dim> temp_rhs;

        /*预分配内存*/
        std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
        std::vector<double> temp_rhs_values(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);

        std::vector<double> phi_t(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_t(dofs_per_cell);

        /*SUPG-PSPG-LSIC稳定化参数*/
        std::vector<double> tauSUPGT(n_q_points);
        StabilizationParameter<dim> stabilization_parameter;
        const double thermal_diffusivity = thermal_cond_fluid / (rho * capacity);

        /*迭代器*/
        auto cell = temp_dof_handler.begin_active();
        const auto endc = temp_dof_handler.end();
        auto ns_cell = ns_dof_handler.begin_active();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++ns_cell, ++top_equivalent_cell)
        {
            if (cell->is_locally_owned())
            {
                temp_fe_values.reinit(cell);
                ns_fe_values.reinit(ns_cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                local_matrix = 0;
                local_rhs = 0;

                ns_fe_values[velocities].get_function_values(ns_solution_relevant,
                                                             present_velocity_values);
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
                    temp_rhs_values[q] = temp_rhs.value(temp_fe_values.quadrature_point(q));
                    thermal_conductivity = material_interpolate.conductivity_interpolate(xphys_equivalent_values_q_point);
                    heatsource_coeff = material_interpolate.heatsource_interpolate(xphys_equivalent_values_q_point);
                    convection_coeff = material_interpolate.convection_interpolate(xphys_equivalent_values_q_point);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        phi_t[k] = temp_fe_values.shape_value(k, q);
                        grad_phi_t[k] = temp_fe_values.shape_grad(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                            local_matrix(i, j) +=
                                (phi_t[i] * (convection_coeff * rho * capacity * (present_velocity_values[q] * grad_phi_t[j]) +
                                             heatsource_coeff * phi_t[j]) +
                                 grad_phi_t[i] * grad_phi_t[j] * thermal_conductivity +
                                 (tauSUPGT[q] * present_velocity_values[q] * grad_phi_t[i]) *
                                     (convection_coeff * rho * capacity * (present_velocity_values[q] * grad_phi_t[j]) +
                                      heatsource_coeff * phi_t[j])) *
                                temp_fe_values.JxW(q);
                        }

                        local_rhs(i) +=
                            ((phi_t[i] + tauSUPGT[q] * (present_velocity_values[q] * grad_phi_t[i])) *
                             (temp_rhs_values[q] + heatsource_coeff * t_Q)) *
                            temp_fe_values.JxW(q);
                    }
                }

                for (const auto face : cell->face_indices())
                {
                    auto face_iter = cell->face(face);
                    if (face_iter->at_boundary() && face_iter->boundary_id() == 4)
                    {
                        temp_fe_face_values.reinit(cell, face);
                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                        {
                            for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                                local_rhs(i) +=
                                    (temp_fe_face_values.shape_value(i, q)) * heatflux0 *
                                    temp_fe_face_values.JxW(q);
                            }
                        }
                    }
                }

                cell->get_dof_indices(local_dof_indices);

                temp_constraints.distribute_local_to_global(local_matrix,
                                                            local_rhs,
                                                            local_dof_indices,
                                                            temp_system_matrix,
                                                            temp_system_rhs);
            }
        }

        temp_system_rhs.compress(VectorOperation::add);

        temp_system_matrix.compress(VectorOperation::add);
    }

    void TopOptHeatFlow::temp_solve()
    {
        TimerOutput::Scope t(computing_timer, "temp_solve");
        /* PETSc Preconditioner*/
        using PreconditionType = PETScWrappers::PreconditionBlockJacobi;
        PreconditionType preconditioner;
        {
            PreconditionType::AdditionalData data;
            preconditioner.initialize(temp_system_matrix, data);
        }

        SolverControl solver_control(temp_system_matrix.m(),
                                     1e-8 * temp_system_rhs.l2_norm(),
                                     true);

        PETScWrappers::SparseDirectMUMPS solver(solver_control);

        pcout << std::endl
              << "******************************************************************************" << std::endl;
        pcout << "Solving For Temperature Field... " << std::endl;

        solver.solve(temp_system_matrix, temp_solution_distributed, temp_system_rhs);

        pcout << "Solver Converged after steps: " << solver_control.last_step() << std::endl;

        temp_constraints.distribute(temp_solution_distributed);
        temp_solution_relevant = temp_solution_distributed;
    }
}