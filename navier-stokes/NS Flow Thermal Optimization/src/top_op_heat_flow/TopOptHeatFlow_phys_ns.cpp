#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*该函数构建了我们当前正在研究的NS系统矩阵和右侧。
    该initial_step参数用于确定我们应用哪一组约束（初始步骤非零，其他步骤为零）。
    该assemble_matrix参数分别决定是组装整个系统还是仅组装右侧向量。*/
    void TopOptHeatFlow::ns_assemble(const bool assemble_matrix)
    {
        TimerOutput::Scope t(computing_timer, "ns_assembly");
        if (assemble_matrix)
            ns_system_matrix = 0;

        ns_system_rhs = 0;

        /*选择迭代方式-newton，oseen，stokes*/
        double pa = 1.0, pb = 1.0;
        switch (iteration_method)
        {
        case PRM::newton_method:
        {
            pa = 1.0;
            pb = 1.0;
            break;
        }
        case PRM::oseen_method:
        {
            pa = 0.0;
            pb = 1.0;
            break;
        }
        case PRM::stokes_method:
        {
            pa = 0.0;
            pb = 0.0;
            break;
        }
        default:
            break;
        }

        QGauss<dim> quadrature_formula(quadrature_degree + 1);

        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values | update_quadrature_points |
                                       update_JxW_values | update_gradients);

        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);

        const unsigned int dofs_per_cell = ns_fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*对于线性化系统，我们为当前速度和梯度以及当前压力创建临时存储。
        实际上，它们都是通过它们在正交点处的形函数获得的。*/

        std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
        std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
        std::vector<SymmetricTensor<2, dim>> present_velocity_symmetric_gradients(n_q_points);

        std::vector<double> present_pressure_values(n_q_points);
        std::vector<Tensor<1, dim>> present_pressure_gradients(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);
        std::vector<double> permeability_values(n_q_points);

        Tensor<1, dim> normal;

        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>> symmgrad_phi_u(dofs_per_cell);
        std::vector<double> phi_p(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

        /*SUPG-PSPG-LSIC稳定化参数*/
        const unsigned int dofs_per_cell_u_scalar = ns_fe.get_sub_fe(0, 1).n_dofs_per_cell();
        std::vector<std::vector<Tensor<1, dim>>>
            grad_phi_u_scalar_q_points(
                n_q_points,
                std::vector<Tensor<1, dim>>(dofs_per_cell_u_scalar,
                                            Tensor<1, dim>()));
        std::vector<double> tauSUPG(n_q_points), tauPSPG(n_q_points), vLSIC(n_q_points);
        StabilizationParameter<dim> stabilization_parameter;
        double kinematic_viscosity = viscosity / rho;

        /*有限元迭代器*/
        auto cell = ns_dof_handler.begin_active();
        auto endc = ns_dof_handler.end();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++top_equivalent_cell)
        {
            if (cell->is_locally_owned())
            {
                ns_fe_values.reinit(cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                local_matrix = 0;
                local_rhs = 0;

                ns_fe_values[velocities].get_function_values(ns_evaluation_relevant,
                                                             present_velocity_values);
                ns_fe_values[velocities].get_function_gradients(
                    ns_evaluation_relevant, present_velocity_gradients);
                ns_fe_values[velocities].get_function_symmetric_gradients(
                    ns_evaluation_relevant, present_velocity_symmetric_gradients);

                ns_fe_values[pressure].get_function_values(ns_evaluation_relevant,
                                                           present_pressure_values);
                ns_fe_values[pressure].get_function_gradients(ns_evaluation_relevant,
                                                              present_pressure_gradients);

                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);
                double element_size = (cell->diameter()) / std::sqrt(2);

                stabilization_parameter.value(present_velocity_values,
                                              kinematic_viscosity, element_size,
                                              tauSUPG, tauPSPG, vLSIC);

                /*SUPG-PSPG-LSIC修正  局部矩阵组装*/
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);

                    permeability = material_interpolate.permeability_interpolate(xphys_equivalent_values_q_point);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        div_phi_u[k] = ns_fe_values[velocities].divergence(k, q);
                        grad_phi_u[k] = ns_fe_values[velocities].gradient(k, q);
                        symmgrad_phi_u[k] = ns_fe_values[velocities].symmetric_gradient(k, q);
                        phi_u[k] = ns_fe_values[velocities].value(k, q);
                        phi_p[k] = ns_fe_values[pressure].value(k, q);
                        grad_phi_p[k] = ns_fe_values[pressure].gradient(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        if (assemble_matrix)
                        {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                local_matrix(i, j) +=
                                    ((phi_u[i]) *
                                         ((rho * (pa * (present_velocity_gradients[q] * phi_u[j]) + pb * (grad_phi_u[j] * present_velocity_values[q]))) +
                                          (permeability * phi_u[j])) +
                                     (2 * viscosity * scalar_product(grad_phi_u[i], symmgrad_phi_u[j])) -
                                     (div_phi_u[i] * phi_p[j]) +
                                     (phi_p[i] * div_phi_u[j]) +
                                     ((tauSUPG[q] * (grad_phi_u[i] * present_velocity_values[q])) + ((1.0 / rho) * tauPSPG[q] * grad_phi_p[i])) *
                                         ((rho * (pa * (present_velocity_gradients[q] * phi_u[j]) + pb * (grad_phi_u[j] * present_velocity_values[q]))) +
                                          grad_phi_p[j] +
                                          (permeability * phi_u[j])) +
                                     (pa * tauSUPG[q] * (grad_phi_u[i] * phi_u[j])) *
                                         ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                          present_pressure_gradients[q] +
                                          (permeability * present_velocity_values[q])) +
                                     (vLSIC[q] * rho * div_phi_u[i] * div_phi_u[j])) *
                                    ns_fe_values.JxW(q);
                            }
                        }

                        double present_velocity_divergence =
                            trace(present_velocity_gradients[q]);

                        local_rhs(i) -=
                            ((phi_u[i]) *
                                 ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                  (permeability * present_velocity_values[q])) +
                             (2 * viscosity * scalar_product(grad_phi_u[i], present_velocity_symmetric_gradients[q])) -
                             (div_phi_u[i] * present_pressure_values[q]) +
                             (phi_p[i] * present_velocity_divergence) +
                             ((tauSUPG[q] * (grad_phi_u[i] * present_velocity_values[q])) + ((1.0 / rho) * tauPSPG[q] * grad_phi_p[i])) *
                                 ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                  present_pressure_gradients[q] +
                                  (permeability * present_velocity_values[q])) +
                             (vLSIC[q] * rho * div_phi_u[i] * present_velocity_divergence)) *
                            ns_fe_values.JxW(q);
                    }
                }

                cell->get_dof_indices(local_dof_indices);

                if (assemble_matrix)
                {
                    ns_zero_constraints.distribute_local_to_global(local_matrix,
                                                                   local_rhs,
                                                                   local_dof_indices,
                                                                   ns_system_matrix,
                                                                   ns_system_rhs);
                }
                else
                {
                    ns_zero_constraints.distribute_local_to_global(local_rhs,
                                                                   local_dof_indices,
                                                                   ns_system_rhs);
                }
            }
        }

        ns_system_rhs.compress(VectorOperation::add);

        if (assemble_matrix)
        {
            ns_system_matrix.compress(VectorOperation::add);
        }
    }

    void TopOptHeatFlow::ns_assemble_system()
    {
        ns_assemble(true);
    }

    void TopOptHeatFlow::ns_assemble_rhs()
    {
        ns_assemble(false);
    }

    /*在此函数中，我们将 FGMRES 与程序开头定义的块预处理器一起使用来求解线性系统。
    这一步我们得到的是解向量。如果这是初始步骤，解向量将为我们提供纳维斯托克斯方程的初始猜测。
    对于初始步骤，应用非零约束以确保满足边界条件。
    在接下来的步骤中，我们将求解牛顿更新，因此使用零约束。*/
    void TopOptHeatFlow::ns_solve()
    {
        TimerOutput::Scope t(computing_timer, "ns_solve");
        /* PETSc Preconditioner*/
        using PreconditionType = PETScWrappers::PreconditionBlockJacobi;
        PreconditionType preconditioner;
        {
            PreconditionType::AdditionalData data;
            preconditioner.initialize(ns_system_matrix, data);
        }

        IterationNumberControl solver_control(ns_system_matrix.m() / 50,
                                              1e-5 * ns_system_rhs.l2_norm(),
                                              true);

        PETScWrappers::SparseDirectMUMPS solver(solver_control);

        solver.solve(ns_system_matrix, ns_newton_update_distributed, ns_system_rhs);
        pcout << "Solver Converged after steps: " << solver_control.last_step() << std::endl;

        ns_zero_constraints.distribute(ns_newton_update_distributed);
        ns_newton_update_relevant = ns_newton_update_distributed;
    }

    void TopOptHeatFlow::ns_nls_assemble(const bool is_jacobian, const LA::MPI::Vector &evaluation_point, LA::MPI::Vector &residual)
    {
        TimerOutput::Scope t(computing_timer, "ns_nls_assembly");
        if (is_jacobian)
        {
            pcout << "Assembling jacobian matrix " << std::endl;
            ns_system_matrix = 0;
        }
        else
        {
            pcout << "Computing residual vector ";
            residual = 0;
        }
        /* 注意，evaluation_point 是NLS算法内部的const变量，未考虑悬挂点约束，
        通过赋值给ns_evaluation_distributed施加约束，并转化为包含ghost元素的ns_evaluation_relevant */
        ns_evaluation_distributed = evaluation_point;
        ns_nonzero_constraints.distribute(ns_evaluation_distributed);
        ns_evaluation_relevant = ns_evaluation_distributed;

        /*选择迭代方式-newton(pa = 1, pb = 1)，oseen，stokes*/
        double pa = 1.0, pb = 1.0;

        QGauss<dim> quadrature_formula(quadrature_degree + 1);

        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values | update_quadrature_points |
                                       update_JxW_values | update_gradients);

        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);

        const unsigned int dofs_per_cell = ns_fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_residual(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*对于线性化系统，我们为当前速度和梯度以及当前压力创建临时存储。
        实际上，它们都是通过它们在正交点处的形函数获得的。*/

        std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
        std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
        std::vector<SymmetricTensor<2, dim>> present_velocity_symmetric_gradients(n_q_points);

        std::vector<double> present_pressure_values(n_q_points);
        std::vector<Tensor<1, dim>> present_pressure_gradients(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);
        std::vector<double> permeability_values(n_q_points);

        Tensor<1, dim> normal;

        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>> symmgrad_phi_u(dofs_per_cell);
        std::vector<double> phi_p(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

        /*SUPG-PSPG-LSIC稳定化参数*/
        const unsigned int dofs_per_cell_u_scalar = ns_fe.get_sub_fe(0, 1).n_dofs_per_cell();
        std::vector<std::vector<Tensor<1, dim>>>
            grad_phi_u_scalar_q_points(
                n_q_points,
                std::vector<Tensor<1, dim>>(dofs_per_cell_u_scalar,
                                            Tensor<1, dim>()));
        std::vector<double> tauSUPG(n_q_points), tauPSPG(n_q_points), vLSIC(n_q_points);
        StabilizationParameter<dim> stabilization_parameter;
        double kinematic_viscosity = viscosity / rho;

        /*有限元迭代器*/
        auto cell = ns_dof_handler.begin_active();
        auto endc = ns_dof_handler.end();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++top_equivalent_cell)
        {
            if (cell->is_locally_owned())
            {
                ns_fe_values.reinit(cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                local_matrix = 0;
                local_residual = 0;

                ns_fe_values[velocities].get_function_values(ns_evaluation_relevant,
                                                             present_velocity_values);
                ns_fe_values[velocities].get_function_gradients(
                    ns_evaluation_relevant, present_velocity_gradients);
                ns_fe_values[velocities].get_function_symmetric_gradients(
                    ns_evaluation_relevant, present_velocity_symmetric_gradients);

                ns_fe_values[pressure].get_function_values(ns_evaluation_relevant,
                                                           present_pressure_values);
                ns_fe_values[pressure].get_function_gradients(ns_evaluation_relevant,
                                                              present_pressure_gradients);

                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);
                double element_size = (cell->diameter()) / std::sqrt(2);

                stabilization_parameter.value(present_velocity_values,
                                              kinematic_viscosity, element_size,
                                              tauSUPG, tauPSPG, vLSIC);

                /*SUPG-PSPG-LSIC修正  局部矩阵组装*/
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);

                    permeability = material_interpolate.permeability_interpolate(xphys_equivalent_values_q_point);

                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        div_phi_u[k] = ns_fe_values[velocities].divergence(k, q);
                        grad_phi_u[k] = ns_fe_values[velocities].gradient(k, q);
                        symmgrad_phi_u[k] = ns_fe_values[velocities].symmetric_gradient(k, q);
                        phi_u[k] = ns_fe_values[velocities].value(k, q);
                        phi_p[k] = ns_fe_values[pressure].value(k, q);
                        grad_phi_p[k] = ns_fe_values[pressure].gradient(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        if (is_jacobian)
                        {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                local_matrix(i, j) +=
                                    ((phi_u[i]) *
                                         ((rho * (pa * (present_velocity_gradients[q] * phi_u[j]) + pb * (grad_phi_u[j] * present_velocity_values[q]))) +
                                          (permeability * phi_u[j])) +
                                     (2 * viscosity * scalar_product(grad_phi_u[i], symmgrad_phi_u[j])) -
                                     (div_phi_u[i] * phi_p[j]) +
                                     (phi_p[i] * div_phi_u[j]) +
                                     ((tauSUPG[q] * (grad_phi_u[i] * present_velocity_values[q])) + ((1.0 / rho) * tauPSPG[q] * grad_phi_p[i])) *
                                         ((rho * (pa * (present_velocity_gradients[q] * phi_u[j]) + pb * (grad_phi_u[j] * present_velocity_values[q]))) +
                                          grad_phi_p[j] +
                                          (permeability * phi_u[j])) +
                                     (pa * tauSUPG[q] * (grad_phi_u[i] * phi_u[j])) *
                                         ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                          present_pressure_gradients[q] +
                                          (permeability * present_velocity_values[q])) +
                                     (vLSIC[q] * rho * div_phi_u[i] * div_phi_u[j])) *
                                    ns_fe_values.JxW(q);
                            }
                        }
                        else
                        {
                            double present_velocity_divergence =
                                trace(present_velocity_gradients[q]);

                            local_residual(i) +=
                                ((phi_u[i]) *
                                     ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                      (permeability * present_velocity_values[q])) +
                                 (2 * viscosity * scalar_product(grad_phi_u[i], present_velocity_symmetric_gradients[q])) -
                                 (div_phi_u[i] * present_pressure_values[q]) +
                                 (phi_p[i] * present_velocity_divergence) +
                                 ((tauSUPG[q] * (grad_phi_u[i] * present_velocity_values[q])) + ((1.0 / rho) * tauPSPG[q] * grad_phi_p[i])) *
                                     ((rho * (present_velocity_gradients[q] * present_velocity_values[q])) +
                                      present_pressure_gradients[q] +
                                      (permeability * present_velocity_values[q])) +
                                 (vLSIC[q] * rho * div_phi_u[i] * present_velocity_divergence)) *
                                ns_fe_values.JxW(q);
                        }
                    }
                }

                cell->get_dof_indices(local_dof_indices);

                if (is_jacobian)
                {
                    ns_zero_constraints.distribute_local_to_global(local_matrix,
                                                                   local_dof_indices,
                                                                   ns_system_matrix);
                }
                else
                {
                    ns_zero_constraints.distribute_local_to_global(local_residual,
                                                                   local_dof_indices,
                                                                   residual);
                }
            }
        }

        if (is_jacobian)
            ns_system_matrix.compress(VectorOperation::add);
        else
        {
            residual.compress(VectorOperation::add);
            pcout << " norm=" << residual.l2_norm() << std::endl;
        }
    }

    void TopOptHeatFlow::ns_nls_assemble_with_residual_linearization(const bool is_jacobian, const LA::MPI::Vector &evaluation_point, LA::MPI::Vector &residual)
    {
        TimerOutput::Scope t(computing_timer, "ns_nls_assembly_with_residual_linearization");
        if (is_jacobian)
        {
            pcout << "Assembling jacobian matrix " << std::endl;
            ns_system_matrix = 0;
        }
        else
        {
            pcout << "Computing residual vector ";
            residual = 0;
        }

        using ADHelper = Differentiation::AD::ResidualLinearization<Differentiation::AD::NumberTypes::sacado_dfad, double>;
        using ADNumberType = typename ADHelper::ad_type;

        /* 注意，evaluation_point 是NLS算法内部的const变量，未考虑悬挂点约束，
        通过赋值给ns_evaluation_distributed施加约束，并转化为包含ghost元素的ns_evaluation_relevant */
        ns_evaluation_distributed = evaluation_point;
        ns_nonzero_constraints.distribute(ns_evaluation_distributed);
        ns_evaluation_relevant = ns_evaluation_distributed;

        QGauss<dim> quadrature_formula(quadrature_degree + 1);

        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        FEValues<dim> ns_fe_values(ns_fe,
                                   quadrature_formula,
                                   update_values | update_quadrature_points |
                                       update_JxW_values | update_gradients);

        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);

        const unsigned int dofs_per_cell = ns_fe.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        const unsigned int n_independent_variables = dofs_per_cell;
        const unsigned int n_dependent_variables = n_independent_variables;
        ADHelper ad_helper(n_independent_variables, n_dependent_variables);

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_residual(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*对于线性化系统，我们为当前速度和梯度以及当前压力创建临时存储。
        实际上，它们都是通过它们在正交点处的形函数获得的。*/

        std::vector<Tensor<1, dim, ADNumberType>> present_velocity_values(n_q_points);
        std::vector<Tensor<2, dim, ADNumberType>> present_velocity_gradients(n_q_points);
        std::vector<SymmetricTensor<2, dim, ADNumberType>> present_velocity_symmetric_gradients(n_q_points);

        std::vector<ADNumberType> present_pressure_values(n_q_points);
        std::vector<Tensor<1, dim, ADNumberType>> present_pressure_gradients(n_q_points);
        std::vector<double> xphys_equivalent_values(n_q_points);
        std::vector<double> permeability_values(n_q_points);

        Tensor<1, dim> normal;

        std::vector<double> div_phi_u(dofs_per_cell);
        std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
        std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
        std::vector<SymmetricTensor<2, dim>> symmgrad_phi_u(dofs_per_cell);
        std::vector<double> phi_p(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);

        /*SUPG-PSPG-LSIC稳定化参数*/
        const unsigned int dofs_per_cell_u_scalar = ns_fe.get_sub_fe(0, 1).n_dofs_per_cell();
        std::vector<std::vector<Tensor<1, dim>>>
            grad_phi_u_scalar_q_points(
                n_q_points,
                std::vector<Tensor<1, dim>>(dofs_per_cell_u_scalar,
                                            Tensor<1, dim>()));
        std::vector<ADNumberType> tauSUPG(n_q_points), tauPSPG(n_q_points), vLSIC(n_q_points);
        StabilizationParameter<dim, ADNumberType> stabilization_parameter;
        double kinematic_viscosity = viscosity / rho;

        /*有限元迭代器*/
        auto cell = ns_dof_handler.begin_active();
        auto endc = ns_dof_handler.end();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++top_equivalent_cell)
        {
            if (cell->is_locally_owned())
            {
                ns_fe_values.reinit(cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                cell->get_dof_indices(local_dof_indices);

                ad_helper.reset(n_independent_variables, n_dependent_variables);
                ad_helper.register_dof_values(ns_evaluation_relevant, local_dof_indices);
                const std::vector<ADNumberType> &dof_values_ad = ad_helper.get_sensitive_dof_values();

                local_matrix = 0;
                local_residual = 0;

                ns_fe_values[velocities].get_function_values_from_local_dof_values(dof_values_ad,
                                                                                   present_velocity_values);
                ns_fe_values[velocities].get_function_gradients_from_local_dof_values(
                    dof_values_ad, present_velocity_gradients);
                ns_fe_values[velocities].get_function_symmetric_gradients_from_local_dof_values(
                    dof_values_ad, present_velocity_symmetric_gradients);

                ns_fe_values[pressure].get_function_values_from_local_dof_values(dof_values_ad,
                                                                                 present_pressure_values);
                ns_fe_values[pressure].get_function_gradients_from_local_dof_values(dof_values_ad,
                                                                                    present_pressure_gradients);

                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant,
                                                                     xphys_equivalent_values);
                double element_size = (cell->diameter()) / std::sqrt(2);

                stabilization_parameter.value(present_velocity_values,
                                              ADNumberType(kinematic_viscosity), ADNumberType(element_size),
                                              tauSUPG, tauPSPG, vLSIC);

                std::vector<ADNumberType> residual_ad(n_dependent_variables, ADNumberType(0.0));
                /*SUPG-PSPG-LSIC修正  局部矩阵组装*/
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    double xphys_equivalent_values_q_point = std::max(std::min(xphys_equivalent_values[q], 1.0), 0.0);

                    permeability = material_interpolate.permeability_interpolate(xphys_equivalent_values_q_point);
                    ADNumberType present_velocity_divergence = trace(present_velocity_gradients[q]);
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        div_phi_u[k] = ns_fe_values[velocities].divergence(k, q);
                        grad_phi_u[k] = ns_fe_values[velocities].gradient(k, q);
                        symmgrad_phi_u[k] = ns_fe_values[velocities].symmetric_gradient(k, q);
                        phi_u[k] = ns_fe_values[velocities].value(k, q);
                        phi_p[k] = ns_fe_values[pressure].value(k, q);
                        grad_phi_p[k] = ns_fe_values[pressure].gradient(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
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
                            ns_fe_values.JxW(q);
                    }
                }

                ad_helper.register_residual_vector(residual_ad);

                if (is_jacobian)
                    ad_helper.compute_linearization(local_matrix);
                else
                    ad_helper.compute_residual(local_residual);

                if (is_jacobian)
                {
                    ns_zero_constraints.distribute_local_to_global(local_matrix,
                                                                   local_dof_indices,
                                                                   ns_system_matrix);
                }
                else
                {
                    ns_zero_constraints.distribute_local_to_global(local_residual,
                                                                   local_dof_indices,
                                                                   residual);
                }
            }
        }

        if (is_jacobian)
            ns_system_matrix.compress(VectorOperation::add);
        else
        {
            residual.compress(VectorOperation::add);
            pcout << " norm=" << residual.l2_norm() << std::endl;
        }
    }

    void TopOptHeatFlow::ns_nls_compute_jacobian(const LA::MPI::Vector &evaluation_point)
    {
        LA::MPI::Vector tmp_vector;
        ns_nls_assemble(true, evaluation_point, tmp_vector);
    }

    void TopOptHeatFlow::ns_nls_compute_residual(const LA::MPI::Vector &evaluation_point, LA::MPI::Vector &residual)
    {
        ns_nls_assemble(false, evaluation_point, residual);
    }

    void TopOptHeatFlow::ns_nls_solve(const LA::MPI::Vector &rhs, LA::MPI::Vector &solution)
    {
        TimerOutput::Scope t(computing_timer, "ns_nls_solve");

        IterationNumberControl solver_control(ns_system_matrix.m() / 50,
                                              1e-5 * rhs.l2_norm(),
                                              true);

        PETScWrappers::SparseDirectMUMPS solver(solver_control);

        solver.solve(ns_system_matrix, solution, rhs);
        pcout << "Solver Converged after steps: " << solver_control.last_step() << std::endl;
        ns_zero_constraints.distribute(solution);
    }

    /*TopOptHeatFlow<dim>::newton_iteration
    该函数使用给定的容差、最大迭代次数以及要执行的网格细化次数来实现牛顿迭代。*/
    void TopOptHeatFlow::newton_iteration(
        const double tolerance,
        const unsigned int max_n_line_searches,
        const unsigned int max_n_refinements)
    {

        pcout << std::endl
              << "******************************************************************************" << std::endl;
        pcout << "Solving For Navier-Stokes Field ";

        /* NonlinearSolverSelector */

        std::string nonlinear_solver_type = "snes";

        if (nonlinear_solver_type == "kinsol")
        {
            /* KINSOL */
            pcout << "with KINSOL..." << std::endl;
            typename SUNDIALS::KINSOL<LA::MPI::Vector>::AdditionalData additional_data;
            additional_data.strategy = SUNDIALS::KINSOL<dealii::PETScWrappers::MPI::Vector>::AdditionalData::linesearch;
            additional_data.function_tolerance = tolerance;
            SUNDIALS::KINSOL<LA::MPI::Vector> nonlinear_solver(additional_data);

            nonlinear_solver.reinit_vector =
                [&](LA::MPI::Vector &x)
            {
                x.reinit(ns_dof_handler.locally_owned_dofs(), mpi_communicator);
            };

            nonlinear_solver.residual =
                [&](const LA::MPI::Vector &evaluation_point, LA::MPI::Vector &residual)
            {
                ns_nls_compute_residual(evaluation_point, residual);
                return 0;
            };

            nonlinear_solver.setup_jacobian =
                [&](const LA::MPI::Vector &evaluation_point, const LA::MPI::Vector & /* f */)
            {
                ns_nls_compute_jacobian(evaluation_point);
                return 0;
            };

            nonlinear_solver.solve_with_jacobian =
                [&](const LA::MPI::Vector &rhs, LA::MPI::Vector &solution, const double tolerance)
            {
                (void)tolerance;
                ns_nls_solve(rhs, solution);
                return 0;
            };

            nonlinear_solver.solve(ns_solution_distributed);
        }
        else if (nonlinear_solver_type == "snes")
        {
            /* SNES */
            pcout << "with SNES..." << std::endl;
            PETScWrappers::NonlinearSolverData addition_data;
            {
                addition_data.options_prefix = "ns_";
                addition_data.snes_type = "newtonls";
                addition_data.snes_linesearch_type = "";

                addition_data.absolute_tolerance = tolerance;
                addition_data.relative_tolerance = 0;
                addition_data.step_tolerance = 0;
                addition_data.maximum_non_linear_iterations = -1;
                addition_data.max_n_function_evaluations = -1;
            }
            PETScWrappers::NonlinearSolver<LA::MPI::Vector, LA::MPI::SparseMatrix> nonlinear_solver(addition_data);

            nonlinear_solver.residual =
                [&](const LA::MPI::Vector &evaluation_point, LA::MPI::Vector &residual)
            {
                ns_nls_compute_residual(evaluation_point, residual);
                return 0;
            };

            bool user_control = true;
            if (user_control)
            {

                nonlinear_solver.setup_jacobian =
                    [&](const LA::MPI::Vector &evaluation_point)
                {
                    ns_nls_compute_jacobian(evaluation_point);
                    return 0;
                };

                nonlinear_solver.solve_with_jacobian =
                    [&](const LA::MPI::Vector &rhs, LA::MPI::Vector &solution)
                {
                    ns_nls_solve(rhs, solution);
                    return 0;
                };

                nonlinear_solver.set_matrix(ns_system_matrix);
            }
            else
            {

                nonlinear_solver.set_matrix(ns_system_matrix);

                nonlinear_solver.jacobian =
                    [&](const LA::MPI::Vector &evaluation_point, LA::MPI::SparseMatrix &, LA::MPI::SparseMatrix &P)
                {
                    Assert(P == ns_system_matrix, ExcInternalError());
                    ns_nls_compute_jacobian(evaluation_point);
                    (void)P;
                    return 0;
                };
            }

            nonlinear_solver.monitor =
                [&](const LA::MPI::Vector &, unsigned int step, double gnorm)
            {
                (void)step;
                (void)gnorm;

                return 0;
            };

            nonlinear_solver.solve(ns_solution_distributed);
        }
        else
        {
            pcout << "with BJH..." << std::endl;
            for (unsigned int refinement_n = 0; refinement_n < max_n_refinements + 1;
                 ++refinement_n)
            {
                unsigned int line_search_n = 0;
                double last_res = std::numeric_limits<double>::max();
                double current_res = std::numeric_limits<double>::max();
                pcout << "grid refinements: " << refinement_n << std::endl
                      << "viscosity: " << viscosity << std::endl;

                while (((current_res > tolerance)) && line_search_n < max_n_line_searches)
                {
                    ns_evaluation_relevant = ns_solution_relevant;
                    ns_assemble_system();
                    ns_solve();

                    /*为了确保我们的解决方案接近精确解决方案，我们使用权重更新解决方案，
                    alpha使得新的残差小于上一步的残差，
                    这是在以下循环中完成的。这与步骤 15中使用的线搜索算法相同。*/
                    for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
                    {
                        ns_evaluation_relevant = ns_solution_relevant;
                        ns_evaluation_distributed = ns_evaluation_relevant;
                        ns_evaluation_distributed.add(alpha, ns_newton_update_relevant);
                        ns_nonzero_constraints.distribute(ns_evaluation_distributed);
                        ns_evaluation_relevant = ns_evaluation_distributed;

                        ns_assemble_rhs();
                        current_res = ns_system_rhs.l2_norm();
                        pcout << "  alpha: " << std::setw(10) << alpha
                              << std::setw(0) << "  residual: " << current_res
                              << std::endl;
                        if (current_res < last_res)
                            break;
                    }
                    {
                        ns_solution_relevant = ns_evaluation_relevant;
                        pcout << "  number of line searches: " << line_search_n
                              << "  residual: " << current_res << std::endl;
                        last_res = current_res;
                    }
                    ++line_search_n;
                }

                ns_solution_distributed = ns_solution_relevant;
            }
        }

        ns_nonzero_constraints.distribute(ns_solution_distributed);
        ns_solution_relevant = ns_solution_distributed;
    }

    /*TopOptHeatFlow::compute_initial_guess
    正如我们在简介中讨论的那样，该函数将通过使用连续方法为我们提供初步猜测。
    雷诺数逐步增加，直到达到目标值。通过实验，斯托克斯的解足以作为雷诺数为 1000 的 NSE 的初始猜测，
    因此我们从这里开始。为了确保上一个问题的解足够接近下一个问题，步长必须足够小*/
    void TopOptHeatFlow::compute_initial_guess(double Re_init, double step_size)
    {
        const double target_viscosity = viscosity;

        for (double Re = Re_init; Re < Re_refer; Re = std::min(Re + step_size, Re_refer))
        {
            viscosity = target_viscosity * Re_refer / Re;
            pcout << std::endl
                  << "******************************************************************************" << std::endl;
            pcout << "Searching for initial guess with Re = " << Re << std::endl;
            newton_iteration(1e-8, 50, 0);
        }
        viscosity = target_viscosity;
        pcout << std::endl
              << "******************************************************************************" << std::endl;
        pcout << "Found initial guess." << std::endl;
        pcout << "Computing solution with target Re = " << Re_refer << std::endl;
    }

    void TopOptHeatFlow::ns_nonlinear_solver(const bool is_init_step)
    {
        const double Re_init = 100.0;
        const double step_size = 100.0;

        if (Re_refer <= Re_init)
        {
            pcout << std::endl
                  << "******************************************************************************" << std::endl;
            pcout << "Re is small enough to be solved directly" << std::endl;
            pcout << "Computing solution with target Re = " << Re_refer << std::endl;

            newton_iteration(1e-8, 50, 0);
        }
        else
        {
            if (is_init_step)
            {

                compute_initial_guess(Re_init, step_size);

                newton_iteration(1e-8, 50, 0);
            }
            else
            {

                try
                {
                    newton_iteration(1e-8, 50, 0);
                }
                catch (const std::exception &exc)
                {
                    pcout << std::endl
                          << "******************************************************************************" << std::endl;
                    pcout << "Navier-Stokes equations did not converge: " << std::endl
                          << exc.what() << std::endl;
                    pcout << "Restart initial guess..." << std::endl;

                    try
                    {
                        compute_initial_guess(Re_init, step_size);

                        newton_iteration(1e-8, 50, 0);
                    }
                    catch (const std::exception &exc)
                    {
                        output_results(1000, 1000);
                        pcout << std::endl
                              << "******************************************************************************" << std::endl;
                        pcout << "Second attempt at nonlinear iteration failed: " << std::endl
                              << exc.what() << std::endl;
                        exit(1);
                    }
                }
            }
        }
    }
}