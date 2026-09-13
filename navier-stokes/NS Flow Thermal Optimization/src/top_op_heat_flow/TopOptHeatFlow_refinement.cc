#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    void TopOptHeatFlow::estimate_topology_boundary(Vector<float> &estimated_error_per_cell)
    {
        TimerOutput::Scope t(computing_timer, "estimate_topology_boundary");

        estimated_error_per_cell = 0.0f;

        QGauss<dim> quadrature_formula(top_equivalent_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);

        const unsigned int n_q_points = quadrature_formula.size();

        /*网格细化水平上下限设置*/
        int level_global = 2;
        int level_boundary_max = 2;
        int level_boundary_min = 2;
        int level_fluid_max = 1;
        int level_fluid_min = 0;
        int level_solid_max = 0;
        int level_solid_min = 0;
        double threshold_solid = 0.05;
        double threshold_fluid = 1.0 - threshold_solid;

        /*预分配内存*/
        std::vector<double> xphys_values(n_q_points);
        double xval;
        float top_bound_estimator;

        /*迭代器*/
        auto cell = top_equivalent_dof_handler.begin_active();
        const auto endc = top_equivalent_dof_handler.end();

        for (unsigned int i = 0; cell != endc; ++i, ++cell)
        {
            top_bound_estimator = 0.0f;

            if (cell->is_locally_owned())
            {
                top_equivalent_fe_values.reinit(cell);

                /*读取高斯点处的函数值*/
                auto &present_top_equivalent_fe_values = top_equivalent_fe_values.get_present_fe_values();
                present_top_equivalent_fe_values.get_function_values(xphys_equivalent_relevant, xphys_values);

                /*目标函数和约束函数计算*/

                for (unsigned int q = 0; q < 1; ++q)
                {
                    xval = std::max(std::min(xphys_values[q], 1.0), 0.0);
                    if ((xval > threshold_solid) && (xval < threshold_fluid))
                    {
                        if ((cell->level()) < level_boundary_min)
                        {
                            cell->set_refine_flag();
                        }
                        else if ((cell->level()) > level_boundary_max)
                        {
                            cell->set_coarsen_flag();
                        }
                        else
                        {
                            if (level_boundary_min == level_boundary_max)
                            {
                                cell->clear_coarsen_flag();
                                cell->clear_refine_flag();
                            }
                            else if ((cell->level()) == level_boundary_min)
                                cell->clear_coarsen_flag();
                            else if ((cell->level()) == level_boundary_max)
                                cell->clear_refine_flag();
                        }
                    }
                    else if (xval >= threshold_fluid)
                    {
                        if ((cell->level()) < level_fluid_min)
                        {
                            cell->set_refine_flag();
                        }
                        else if ((cell->level()) > level_fluid_max)
                        {
                            cell->set_coarsen_flag();
                        }
                        else
                        {
                            if (level_fluid_min == level_fluid_max)
                            {
                                cell->clear_coarsen_flag();
                                cell->clear_refine_flag();
                            }
                            else if ((cell->level()) == level_fluid_min)
                                cell->clear_coarsen_flag();
                            else if ((cell->level()) == level_fluid_max)
                                cell->clear_refine_flag();
                        }
                    }
                    else
                    {
                        if ((cell->level()) < level_solid_min)
                        {
                            cell->set_refine_flag();
                        }
                        else if ((cell->level()) > level_solid_max)
                        {
                            cell->set_coarsen_flag();
                        }
                        else
                        {
                            if (level_solid_min == level_solid_max)
                            {
                                cell->clear_coarsen_flag();
                                cell->clear_refine_flag();
                            }
                            else if ((cell->level()) == level_solid_min)
                                cell->clear_coarsen_flag();
                            else if ((cell->level()) == level_solid_max)
                                cell->clear_refine_flag();
                        }
                    }

                    if (cell->level() == level_global)
                    {
                        cell->clear_refine_flag();
                    }
                }
                estimated_error_per_cell(i) = top_bound_estimator;
            }
        }
    }

    /*TopOptHeatFlow::refine_mesh
    在对粗网格找到良好的初始猜测后，我们希望通过细化网格来减少误差。
    这里我们进行类似于步骤 15 的自适应细化，只是我们仅在速度上使用凯利估计器。
    我们还需要使用SolutionTransfer类将当前解决方案传输到下一个网格。*/
    void TopOptHeatFlow::refine_mesh()
    {
        TimerOutput::Scope t(computing_timer, "refine_mesh");
        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

        const FEValuesExtractors::Vector velocity(0);
        KellyErrorEstimator<dim>::estimate(
            ns_dof_handler,
            QGauss<dim - 1>(degree + 1),
            std::map<types::boundary_id, const Function<dim> *>(),
            ns_solution_relevant,
            estimated_error_per_cell,
            ns_fe.component_mask(velocity));

        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
            triangulation,
            estimated_error_per_cell,
            0.3,
            0.03);

        estimate_topology_boundary(estimated_error_per_cell);

        const std::vector<const LA::MPI::Vector *>
            vectors_top_in = {&xphys_values_relevant,
                              &xold1_relevant,
                              &xold2_relevant,
                              &U_relevant,
                              &L_relevant};

        triangulation.prepare_coarsening_and_refinement();

        parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> solution_transfer_ns(ns_dof_handler, false);
        solution_transfer_ns.prepare_for_coarsening_and_refinement(ns_solution_relevant);
        parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> solution_transfer_top(top_dof_handler, false);
        solution_transfer_top.prepare_for_coarsening_and_refinement(vectors_top_in);

        triangulation.execute_coarsening_and_refinement();
        define_grid();
        get_grid_information();

        /*首先设置DoFHandler并生成约束。然后我们创建一个临时的Vector tmp，
        其大小与新网格上的解一致。*/
        setup_dofs();

        LA::MPI::Vector tmp_ns(ns_owned_partitioning, mpi_communicator);

        LA::MPI::Vector tmp_xphys(top_owned_partitioning, mpi_communicator);
        LA::MPI::Vector tmp_xold1(top_owned_partitioning, mpi_communicator);
        LA::MPI::Vector tmp_xold2(top_owned_partitioning, mpi_communicator);
        LA::MPI::Vector tmp_U(top_owned_partitioning, mpi_communicator);
        LA::MPI::Vector tmp_L(top_owned_partitioning, mpi_communicator);

        std::vector<LA::MPI::Vector *> vectors_top_out = {&tmp_xphys, &tmp_xold1, &tmp_xold2, &tmp_U, &tmp_L};

        /*将解从粗网格转移到细网格，并对新转移的解应用边界值约束。
        请注意，present_solution 仍然是与旧网格对应的向量。*/
        solution_transfer_ns.interpolate(tmp_ns);
        ns_nonzero_constraints.distribute(tmp_ns);

        solution_transfer_top.interpolate(vectors_top_out);
        top_constraints.distribute(tmp_xphys);
        top_constraints.distribute(tmp_xold1);
        top_constraints.distribute(tmp_xold2);
        top_constraints.distribute(tmp_U);
        top_constraints.distribute(tmp_L);

        /*最后重置矩阵和向量并将present_solution设置为插值数据。*/
        initialize_system();

        ns_solution_distributed = tmp_ns;
        ns_solution_relevant = ns_solution_distributed;

        xphys_values_distributed = tmp_xphys;
        xphys_values_relevant = xphys_values_distributed;
        xold1_distributed = tmp_xold1;
        xold1_relevant = xold1_distributed;
        xold2_distributed = tmp_xold2;
        xold2_relevant = xold2_distributed;
        U_distributed = tmp_U;
        U_relevant = U_distributed;
        L_distributed = tmp_L;
        L_relevant = L_distributed;
    }

}