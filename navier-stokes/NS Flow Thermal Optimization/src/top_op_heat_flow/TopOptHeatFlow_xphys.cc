#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{
    /*初始化伪密度-及非设计域元素置0或置1*/
    void TopOptHeatFlow::reset_topology(const bool is_initial_step)
    {
        TimerOutput::Scope t(computing_timer, "reset_topology");

        if (is_initial_step)
        {
            xphys_values_distributed = start_point;
            xphys_values_relevant = xphys_values_distributed;
        }

        Vector<double> local_values(1);

        auto top_cell = top_dof_handler.begin_active();
        const auto endc = top_dof_handler.end();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();

        for (; top_cell != endc; top_cell++, top_equivalent_cell++)
        {
            if (top_cell->is_locally_owned() && top_cell->material_id() != 0)
            {
                if (top_cell->material_id() == 1)
                {
                    local_values = 1.0;
                }
                else if (top_cell->material_id() == 2)
                {
                    local_values = 0.0;
                }
                top_cell->set_dof_values(local_values, xphys_values_distributed);
                top_cell->set_dof_values(local_values, xphys_filter_distributed);
                top_cell->set_dof_values(local_values, xphys_heaviside_distributed);
                top_equivalent_cell->set_dof_values(local_values, xphys_equivalent_distributed);
            }
        }
        xphys_values_distributed.compress(VectorOperation::insert);
        xphys_filter_distributed.compress(VectorOperation::insert);
        xphys_heaviside_distributed.compress(VectorOperation::insert);
        xphys_equivalent_distributed.compress(VectorOperation::insert);
        top_constraints.distribute(xphys_values_distributed);
        top_constraints.distribute(xphys_filter_distributed);
        top_constraints.distribute(xphys_heaviside_distributed);
        top_equivalent_constraints.distribute(xphys_equivalent_distributed);
        xphys_values_relevant = xphys_values_distributed;
        xphys_filter_relevant = xphys_filter_distributed;
        xphys_heaviside_relevant = xphys_heaviside_distributed;
        xphys_equivalent_relevant = xphys_equivalent_distributed;
    }

    /*拓扑变量投影*/
    void TopOptHeatFlow::xphys_heaviside()
    {
        TimerOutput::Scope t(computing_timer, "xphys_heaviside");

        QGauss<dim> quadrature_formula(top_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values);

        const unsigned int dofs_per_cell = top_dof_handler.get_fe_collection()[0].n_dofs_per_cell();

        Vector<double> local_top_values(dofs_per_cell);

        /*预分配内存*/
        Vector<double> xphys_filter_dof_values(dofs_per_cell);

        /*迭代器*/
        auto cell = top_dof_handler.begin_active();
        const auto endc = top_dof_handler.end();

        for (; cell != endc; ++cell)
        {

            if (cell->is_locally_owned() && cell->material_id() == 0)
            {
                local_top_values = 0;
                top_fe_values.reinit(cell);

                cell->get_dof_values(xphys_filter_relevant,
                                     xphys_filter_dof_values);

                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    local_top_values(k) =
                        std::max(std::min(heaviside.value(xphys_filter_dof_values(k)),
                                          1.0),
                                 0.0);
                }

                cell->set_dof_values(local_top_values, xphys_heaviside_distributed);
            }
        }

        xphys_heaviside_distributed.compress(VectorOperation::insert);

        top_constraints.distribute(xphys_heaviside_distributed);

        xphys_heaviside_relevant = xphys_heaviside_distributed;

        xphys_range_constraint(xphys_heaviside_distributed,
                               xphys_heaviside_relevant);
    }

    void TopOptHeatFlow::xphys_heaviside_conservation()
    {

        xphys_heaviside();
    }

    void TopOptHeatFlow::xphys_equivalent()
    {
        TimerOutput::Scope t(computing_timer, "xphys_equivalent");

        QGauss<dim> quadrature_formula(top_degree + 1);
        hp::QCollection<dim> hp_quadrature_formula(quadrature_formula);

        hp::FEValues<dim> top_fe_values(top_mapping,
                                        top_dof_handler.get_fe_collection(),
                                        hp_quadrature_formula,
                                        update_values | update_JxW_values);
        hp::FEValues<dim> top_equivalent_fe_values(top_equivalent_mapping,
                                                   top_equivalent_fe,
                                                   hp_quadrature_formula,
                                                   update_values);

        const unsigned int top_n_q_points = quadrature_formula.size();
        const unsigned int top_equivalent_n_dofs_per_cell = top_equivalent_fe[0].n_dofs_per_cell();

        Vector<double> local_top_equivalent_values(top_equivalent_n_dofs_per_cell);
        std::vector<double> xphys_values(top_n_q_points);

        /*预分配内存*/
        double local_top_values = 0;
        double local_volume = 0;

        /*迭代器*/
        auto top_cell = top_dof_handler.begin_active();
        auto top_equivalent_cell = top_equivalent_dof_handler.begin_active();
        const auto endc = top_dof_handler.end();

        for (; top_cell != endc; ++top_cell, ++top_equivalent_cell)
        {

            if (top_cell->is_locally_owned() && top_cell->material_id() == 0)
            {
                top_fe_values.reinit(top_cell);
                top_equivalent_fe_values.reinit(top_equivalent_cell);

                if (top_degree == 0)
                {
                    top_cell->get_dof_values(xphys_heaviside_relevant, local_top_equivalent_values);
                    local_top_values = std::max(std::min(local_top_equivalent_values[0],
                                                         1.0),
                                                0.0);

                    local_top_values = heaviside_equivalent.value(local_top_values);

                    local_top_equivalent_values = local_top_values;
                }
                else
                {
                    local_top_equivalent_values = 0;
                    local_top_values = 0;
                    local_volume = top_cell->measure();

                    auto &present_top_fe_values = top_fe_values.get_present_fe_values();
                    present_top_fe_values.get_function_values(xphys_heaviside_relevant,
                                                              xphys_values);

                    for (unsigned int q = 0; q < top_n_q_points; ++q)
                    {
                        local_top_values += xphys_values[q] * present_top_fe_values.JxW(q);
                    }

                    local_top_values = std::max(std::min(local_top_values / local_volume,
                                                         1.0),
                                                0.0);

                    local_top_values = heaviside_equivalent.value(local_top_values);

                    local_top_equivalent_values = local_top_values;
                }

                top_equivalent_cell->set_dof_values(local_top_equivalent_values, xphys_equivalent_distributed);
            }
        }

        xphys_equivalent_distributed.compress(VectorOperation::insert);

        top_equivalent_constraints.distribute(xphys_equivalent_distributed);

        xphys_equivalent_relevant = xphys_equivalent_distributed;

        xphys_range_constraint(xphys_equivalent_distributed,
                               xphys_equivalent_relevant);
    }

    void TopOptHeatFlow::xphys_range_constraint(LA::MPI::Vector &top_vector_distributed,
                                                LA::MPI::Vector &top_vector_relevant)
    {
        (void)top_vector_distributed;
        (void)top_vector_relevant;
    }

}