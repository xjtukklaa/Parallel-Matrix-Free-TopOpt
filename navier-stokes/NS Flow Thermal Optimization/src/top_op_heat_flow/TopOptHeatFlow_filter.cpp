#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*过滤器系统和右端项组装*/
    void TopOptHeatFlow::filter_assemble(const bool assemble_matrix,
                                         const LA::MPI::Vector &top_vector_relevant)
    {
        TimerOutput::Scope t(computing_timer, "filter_assembly");
        if (assemble_matrix)
        {
            filter_system_matrix = 0;
        }
        filter_system_rhs = 0;

        QGauss<dim> quadrature_formula(filter_fe[0].degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values);

        hp::FEValues<dim> filter_fe_values(filter_mapping,
                                           filter_fe,
                                           hp_quadrature_formula,
                                           update_values | update_quadrature_points |
                                               update_JxW_values | update_gradients);

        const unsigned int dofs_per_cell = filter_fe[0].n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
        Vector<double> local_rhs(dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        /*Helmholtz过滤系数*/

        double coeff = filter_radius * filter_radius / 12.0;

        /*预分配内存*/
        std::vector<double> present_top_values(n_q_points);

        std::vector<double> phi_t(dofs_per_cell);
        std::vector<Tensor<1, dim>> grad_phi_t(dofs_per_cell);

        /*迭代器*/
        auto cell = filter_dof_handler.begin_active();
        const auto endc = filter_dof_handler.end();
        auto top_cell = top_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++top_cell)
        {
            if (cell->is_locally_owned() && cell->material_id() == 0)
            {
                filter_fe_values.reinit(cell);
                top_fe_values.reinit(top_cell);

                local_matrix = 0;
                local_rhs = 0;

                auto &present_filter_fe_values = filter_fe_values.get_present_fe_values();
                auto &present_top_fe_values = top_fe_values.get_present_fe_values();
                present_top_fe_values.get_function_values(top_vector_relevant,
                                                          present_top_values);

                /*SUPG-PSPG-LSIC修正  局部矩阵组装*/
                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        phi_t[k] = present_filter_fe_values.shape_value(k, q);
                        grad_phi_t[k] = present_filter_fe_values.shape_grad(k, q);
                    }

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        if (assemble_matrix)
                        {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                                local_matrix(i, j) +=
                                    (grad_phi_t[i] * grad_phi_t[j] * coeff +
                                     phi_t[i] * phi_t[j]) *
                                    present_filter_fe_values.JxW(q);
                            }
                        }
                        local_rhs(i) +=
                            (phi_t[i] * present_top_values[q]) *
                            present_filter_fe_values.JxW(q);
                    }
                }

                {
                }

                cell->get_dof_indices(local_dof_indices);

                if (assemble_matrix)
                {
                    filter_constraints.distribute_local_to_global(local_matrix,
                                                                  local_rhs,
                                                                  local_dof_indices,
                                                                  filter_system_matrix,
                                                                  filter_system_rhs);
                }
                else
                {
                    filter_constraints.distribute_local_to_global(local_rhs,
                                                                  local_dof_indices,
                                                                  filter_system_rhs);
                }
            }
        }

        if (assemble_matrix)
        {
            filter_system_matrix.compress(VectorOperation::add);
        }
        filter_system_rhs.compress(VectorOperation::add);
    }

    /*过滤求解*/
    void TopOptHeatFlow::filter_solve()
    {
        TimerOutput::Scope t(computing_timer, "filter_solve");
        /* PETSc Preconditioner*/
        using PreconditionType = PETScWrappers::PreconditionJacobi;
        PreconditionType preconditioner;
        {
            PreconditionType::AdditionalData data;
            preconditioner.initialize(filter_system_matrix, data);
        }

        SolverControl solver_control(filter_system_matrix.m(),
                                     1e-12 * filter_system_rhs.l2_norm(),
                                     true);

        PETScWrappers::SolverCG solver(solver_control);

        pcout << std::endl
              << "******************************************************************************" << std::endl;
        pcout << "Solving For Helmholtz Filter... " << std::endl;

        solver.solve(filter_system_matrix, filter_solution_distributed, filter_system_rhs, preconditioner);

        pcout << "Solver Converged after steps: " << solver_control.last_step() << std::endl;

        filter_constraints.distribute(filter_solution_distributed);
        filter_solution_relevant = filter_solution_distributed;
    }

    /*高阶单元向拓扑变量映射，主要用于过滤后的变量向拓扑变量映射*/
    void TopOptHeatFlow::filter_transfer_to_top(const bool is_sens,
                                                LA::MPI::Vector &top_vector_distributed,
                                                LA::MPI::Vector &top_vector_relevant)
    {
        (void)is_sens;
        TimerOutput::Scope t(computing_timer, "filter_transfer_to_top");

        QGauss<dim> quadrature_formula(filter_fe[0].degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        hp::FEValues<dim> filter_fe_values(filter_mapping,
                                           filter_fe,
                                           hp_quadrature_formula,
                                           update_values | update_JxW_values);
        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values);

        const unsigned int dofs_per_cell = top_dof_handler.get_fe_collection()[0].n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        Vector<double> local_filter_values(dofs_per_cell);
        Vector<double> local_top_values(dofs_per_cell);

        /*预分配内存*/
        std::vector<double> present_filter_values(n_q_points);

        /*迭代器*/
        auto cell = top_dof_handler.begin_active();
        const auto endc = top_dof_handler.end();
        auto filter_cell = filter_dof_handler.begin_active();

        for (; cell != endc; ++cell, ++filter_cell)
        {

            if (cell->is_locally_owned() && cell->material_id() == 0)
            {
                local_top_values = 0;

                top_fe_values.reinit(cell);
                filter_fe_values.reinit(filter_cell);

                if (top_degree == 0)
                {

                    auto &present_filter_fe_values = filter_fe_values.get_present_fe_values();
                    present_filter_fe_values.get_function_values(filter_solution_relevant,
                                                                 present_filter_values);

                    double element_volume = cell->measure();

                    const unsigned int k = 0;
                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        local_top_values(k) +=
                            present_filter_values[q] * present_filter_fe_values.JxW(q);
                    }
                    local_top_values(k) = local_top_values(k) / element_volume;
                }
                else
                {

                    filter_cell->get_dof_values(filter_solution_relevant, local_filter_values);
                    local_top_values = local_filter_values;
                }
                cell->set_dof_values(local_top_values, top_vector_distributed);
            }
        }

        top_vector_distributed.compress(VectorOperation::insert);

        top_constraints.distribute(top_vector_distributed);

        top_vector_relevant = top_vector_distributed;

        /* 测试结果表明：
        1）deal.ii向量与Petsc向量完全一致；
        2）Fe_Nothing有限元定义的单元上不存在自由度，这表现为向量仅在非Fe_Nothing单元上有自由度和值
        End */
    }

}