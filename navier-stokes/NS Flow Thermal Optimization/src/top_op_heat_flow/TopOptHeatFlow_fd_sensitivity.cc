

#include "../../include/top_opt_heat_flow.h"

#include <algorithm>

#include <fstream>

#include <iomanip>

#include <vector>

namespace TopOpt
{

    namespace
    {

        struct FDCell
        {

            unsigned long long dof;

            double x;

            double y;
        };

        struct FDValues
        {

            double objective;

            double flow_constraint;

            double volume_constraint;
        };

        double fd_relative_error(const double finite_difference,
                                 const double adjoint)
        {

            const double denominator =
                std::max({1e-12,
                          std::abs(finite_difference),
                          std::abs(adjoint)});

            return std::abs(finite_difference - adjoint) / denominator;
        }
    }

    void TopOptHeatFlow::finite_difference_sensitivity_check(
        const double x1,
        const double x2,
        const double y1,
        const double y2,
        const double perturbation,
        const unsigned int max_cells)
    {

        TimerOutput::Scope timer(computing_timer,
                                 "finite_difference_sensitivity");

        AssertThrow(top_degree == 0,
                    ExcMessage("The finite-difference checker requires "
                               "one DG0 design DoF per cell."));

        AssertThrow(perturbation > 0.0,
                    ExcMessage("FD perturbation must be positive."));

        const double coordinate_scale =
            (PRM::dimensionless_method == PRM::dimensionless_on) ? L_refer : 1.0;

        std::vector<unsigned long long> local_dofs;
        std::vector<double> local_x;
        std::vector<double> local_y;

        for (const auto &cell : top_dof_handler.active_cell_iterators())
        {

            if (!cell->is_locally_owned() || cell->material_id() != 0)
                continue;

            const double center_x = cell->center()[0] * coordinate_scale;
            const double center_y = cell->center()[1] * coordinate_scale;

            if (center_x < x1 || center_x > x2 ||
                center_y < y1 || center_y > y2)
                continue;

            std::vector<types::global_dof_index> dof_indices(
                cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(dof_indices);

            AssertThrow(dof_indices.size() == 1,
                        ExcMessage("DG0 finite-difference checking expects "
                                   "one design DoF per cell."));

            local_dofs.push_back(
                static_cast<unsigned long long>(dof_indices[0]));
            local_x.push_back(center_x);
            local_y.push_back(center_y);
        }

        const int n_processes = static_cast<int>(
            Utilities::MPI::n_mpi_processes(mpi_communicator));
        const int local_count = static_cast<int>(local_dofs.size());

        std::vector<int> counts(n_processes, 0);
        std::vector<int> displacements(n_processes, 0);

        MPI_Allgather(&local_count, 1, MPI_INT,
                      counts.data(), 1, MPI_INT,
                      mpi_communicator);

        int total_count = 0;
        for (int rank = 0; rank < n_processes; ++rank)
        {
            displacements[rank] = total_count;
            total_count += counts[rank];
        }

        if (total_count == 0)
        {
            pcout << "No design cells were found in the FD region." << std::endl;
            return;
        }

        std::vector<unsigned long long> global_dofs(total_count);
        std::vector<double> global_x(total_count);
        std::vector<double> global_y(total_count);

        MPI_Allgatherv(local_dofs.data(), local_count, MPI_UNSIGNED_LONG_LONG,
                       global_dofs.data(), counts.data(), displacements.data(),
                       MPI_UNSIGNED_LONG_LONG, mpi_communicator);
        MPI_Allgatherv(local_x.data(), local_count, MPI_DOUBLE,
                       global_x.data(), counts.data(), displacements.data(),
                       MPI_DOUBLE, mpi_communicator);
        MPI_Allgatherv(local_y.data(), local_count, MPI_DOUBLE,
                       global_y.data(), counts.data(), displacements.data(),
                       MPI_DOUBLE, mpi_communicator);

        std::vector<FDCell> selected_cells;
        selected_cells.reserve(total_count);
        for (int i = 0; i < total_count; ++i)
            selected_cells.push_back({global_dofs[i], global_x[i], global_y[i]});

        std::sort(selected_cells.begin(), selected_cells.end(),
                  [](const FDCell &left, const FDCell &right)
                  { return left.dof < right.dof; });

        if (max_cells > 0 && selected_cells.size() > max_cells)
            selected_cells.resize(max_cells);

        pcout << "Finite-difference checking " << selected_cells.size()
              << " design cells." << std::endl;

        LA::MPI::Vector xphys_base;
        xphys_base.reinit(top_owned_partitioning, mpi_communicator);
        xphys_base = xphys_values_distributed;

        LA::MPI::Vector ns_base;
        ns_base.reinit(ns_owned_partitioning, mpi_communicator);
        ns_base = ns_solution_distributed;

        LA::MPI::Vector temperature_base;
        temperature_base.reinit(temp_owned_partitioning, mpi_communicator);
        temperature_base = temp_solution_distributed;

        LA::MPI::Vector selected_distributed;
        LA::MPI::Vector fd_obj_distributed;
        LA::MPI::Vector fd_flow_distributed;
        LA::MPI::Vector fd_vol_distributed;

        selected_distributed.reinit(top_owned_partitioning, mpi_communicator);
        fd_obj_distributed.reinit(top_owned_partitioning, mpi_communicator);
        fd_flow_distributed.reinit(top_owned_partitioning, mpi_communicator);
        fd_vol_distributed.reinit(top_owned_partitioning, mpi_communicator);
        selected_distributed = 0.0;
        fd_obj_distributed = 0.0;
        fd_flow_distributed = 0.0;
        fd_vol_distributed = 0.0;

        const auto read_owned_value =
            [&](const LA::MPI::Vector &vector,
                const types::global_dof_index dof)
        {
            double local_value = 0.0;

            if (top_owned_partitioning.is_element(dof))
                local_value = vector[dof];

            return Utilities::MPI::sum(local_value, mpi_communicator);
        };

        const auto evaluate =
            [&](const types::global_dof_index dof,
                const double perturbed_design_value)
        {
            xphys_values_distributed = xphys_base;

            if (top_owned_partitioning.is_element(dof))
                xphys_values_distributed[dof] = perturbed_design_value;

            xphys_values_distributed.compress(VectorOperation::insert);
            top_constraints.distribute(xphys_values_distributed);
            xphys_values_relevant = xphys_values_distributed;

            filter_assemble(false, xphys_values_relevant);
            filter_solve();
            filter_transfer_to_top(false,
                                   xphys_filter_distributed,
                                   xphys_filter_relevant);
            xphys_heaviside_conservation();
            xphys_equivalent();

            ns_solution_distributed = ns_base;
            ns_solution_relevant = ns_solution_distributed;
            ns_nonlinear_solver(false);

            temp_assemble();
            temp_solve();
            update_function_values(false);

            return FDValues{-obj_value, cstr_flow_value, cstr_vol_value};
        };

        std::ofstream csv_output;
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {

            std::ostringstream oss;
            oss << std::scientific << std::setprecision(1)
                << perturbation;
            const std::string perturbation_str = oss.str();
            const std::string filename =
                "fd_sensitivity_loop_" +
                Utilities::int_to_string(loop_counter, 4) +
                "_h_" + perturbation_str + ".csv";
            csv_output.open(filename);

            csv_output << "dof,x_m,y_m,x0,x_minus,x_plus,"
                       << "fd_obj,adj_obj,relerr_obj,"
                       << "fd_flow,adj_flow,relerr_flow,"
                       << "fd_volume,adj_volume,relerr_volume\n";
            csv_output << std::setprecision(16);
        }

        for (const FDCell &selected_cell : selected_cells)

        {
            const auto dof = static_cast<types::global_dof_index>(

                selected_cell.dof);
            const double x0 = read_owned_value(xphys_base, dof);

            const double x_minus = std::max(0.0, x0 - perturbation);

            const double x_plus = std::min(1.0, x0 + perturbation);
            const double actual_step = x_plus - x_minus;

            AssertThrow(actual_step > 0.0,
                        ExcMessage("Unable to perturb selected design variable."));

            const FDValues minus_values = evaluate(dof, x_minus);
            const FDValues plus_values = evaluate(dof, x_plus);

            const double fd_objective =
                (plus_values.objective - minus_values.objective) / actual_step;

            const double fd_flow =
                (plus_values.flow_constraint - minus_values.flow_constraint) /
                actual_step;
            const double fd_volume =
                (plus_values.volume_constraint - minus_values.volume_constraint) /
                actual_step;

            const double adjoint_objective =
                read_owned_value(sens_obj_dfilter_distributed, dof);
            const double adjoint_flow =
                read_owned_value(sens_flow_dfilter_distributed, dof);
            const double adjoint_volume =
                read_owned_value(sens_vol_dfilter_distributed, dof);

            if (top_owned_partitioning.is_element(dof))
            {
                selected_distributed[dof] = 1.0;
                fd_obj_distributed[dof] = fd_objective;
                fd_flow_distributed[dof] = fd_flow;
                fd_vol_distributed[dof] = fd_volume;
            }

            if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
                csv_output << selected_cell.dof << ','
                           << selected_cell.x << ',' << selected_cell.y << ','
                           << x0 << ',' << x_minus << ',' << x_plus << ','
                           << fd_objective << ',' << adjoint_objective << ','
                           << fd_relative_error(fd_objective, adjoint_objective) << ','
                           << fd_flow << ',' << adjoint_flow << ','
                           << fd_relative_error(fd_flow, adjoint_flow) << ','
                           << fd_volume << ',' << adjoint_volume << ','
                           << fd_relative_error(fd_volume, adjoint_volume) << '\n';
            }
        }

        selected_distributed.compress(VectorOperation::insert);
        fd_obj_distributed.compress(VectorOperation::insert);
        fd_flow_distributed.compress(VectorOperation::insert);
        fd_vol_distributed.compress(VectorOperation::insert);

        if (csv_output.is_open())
            csv_output.close();

        xphys_values_distributed = xphys_base;
        xphys_values_relevant = xphys_values_distributed;
        filter_assemble(false, xphys_values_relevant);
        filter_solve();
        filter_transfer_to_top(false,
                               xphys_filter_distributed,
                               xphys_filter_relevant);
        xphys_heaviside_conservation();
        xphys_equivalent();

        ns_solution_distributed = ns_base;
        ns_solution_relevant = ns_solution_distributed;
        temp_solution_distributed = temperature_base;
        temp_solution_relevant = temp_solution_distributed;
        update_function_values(false);

        output_finite_difference_sensitivity(selected_distributed,
                                             fd_obj_distributed,
                                             fd_flow_distributed,
                                             fd_vol_distributed,
                                             loop_counter,
                                             perturbation);

        pcout << "Finite-difference sensitivity check finished." << std::endl;
    }

    void TopOptHeatFlow::output_finite_difference_sensitivity(
        const LA::MPI::Vector &selected_distributed,
        const LA::MPI::Vector &fd_obj_distributed,
        const LA::MPI::Vector &fd_flow_distributed,
        const LA::MPI::Vector &fd_vol_distributed,
        const unsigned int loop_num,
        const double perturbation)
    {

        TimerOutput::Scope timer(computing_timer,
                                 "output_fd_sensitivity");

        LA::MPI::Vector selected_relevant;
        LA::MPI::Vector fd_obj_relevant;
        LA::MPI::Vector fd_flow_relevant;
        LA::MPI::Vector fd_vol_relevant;

        selected_relevant.reinit(top_owned_partitioning,
                                 top_relevant_partitioning,
                                 mpi_communicator);
        fd_obj_relevant.reinit(top_owned_partitioning,
                               top_relevant_partitioning,
                               mpi_communicator);
        fd_flow_relevant.reinit(top_owned_partitioning,
                                top_relevant_partitioning,
                                mpi_communicator);
        fd_vol_relevant.reinit(top_owned_partitioning,
                               top_relevant_partitioning,
                               mpi_communicator);

        selected_relevant = selected_distributed;
        fd_obj_relevant = fd_obj_distributed;
        fd_flow_relevant = fd_flow_distributed;
        fd_vol_relevant = fd_vol_distributed;

        LA::MPI::Vector adj_obj_distributed;
        LA::MPI::Vector adj_flow_distributed;
        LA::MPI::Vector adj_vol_distributed;
        LA::MPI::Vector abs_error_obj_distributed;
        LA::MPI::Vector abs_error_flow_distributed;
        LA::MPI::Vector abs_error_vol_distributed;
        LA::MPI::Vector rel_error_obj_distributed;
        LA::MPI::Vector rel_error_flow_distributed;
        LA::MPI::Vector rel_error_vol_distributed;

        std::vector<LA::MPI::Vector *> output_owned_vectors = {
            &adj_obj_distributed, &adj_flow_distributed, &adj_vol_distributed,
            &abs_error_obj_distributed, &abs_error_flow_distributed,
            &abs_error_vol_distributed, &rel_error_obj_distributed,
            &rel_error_flow_distributed, &rel_error_vol_distributed};

        for (auto *vector : output_owned_vectors)
        {
            vector->reinit(top_owned_partitioning, mpi_communicator);
            *vector = 0.0;
        }

        const std::pair<types::global_dof_index, types::global_dof_index>
            owned_range = selected_distributed.local_range();
        for (types::global_dof_index i = owned_range.first;
             i < owned_range.second; ++i)
        {

            if (selected_distributed[i] < 0.5)
                continue;

            const double fd_obj = fd_obj_distributed[i];
            const double fd_flow = fd_flow_distributed[i];
            const double fd_vol = fd_vol_distributed[i];
            const double adj_obj = sens_obj_dfilter_relevant[i];
            const double adj_flow = sens_flow_dfilter_relevant[i];
            const double adj_vol = sens_vol_dfilter_relevant[i];

            adj_obj_distributed[i] = adj_obj;
            adj_flow_distributed[i] = adj_flow;
            adj_vol_distributed[i] = adj_vol;

            abs_error_obj_distributed[i] = std::abs(fd_obj - adj_obj);
            abs_error_flow_distributed[i] = std::abs(fd_flow - adj_flow);
            abs_error_vol_distributed[i] = std::abs(fd_vol - adj_vol);

            rel_error_obj_distributed[i] = fd_relative_error(fd_obj, adj_obj);
            rel_error_flow_distributed[i] = fd_relative_error(fd_flow, adj_flow);
            rel_error_vol_distributed[i] = fd_relative_error(fd_vol, adj_vol);
        }

        for (auto *vector : output_owned_vectors)
            vector->compress(VectorOperation::insert);

        std::vector<LA::MPI::Vector> output_relevant_vectors(9);
        for (unsigned int i = 0; i < output_relevant_vectors.size(); ++i)
        {
            output_relevant_vectors[i].reinit(top_owned_partitioning,
                                              top_relevant_partitioning,
                                              mpi_communicator);
            output_relevant_vectors[i] = *output_owned_vectors[i];
        }

        DataOut<dim> data_out;
        data_out.attach_dof_handler(top_dof_handler);

        data_out.add_data_vector(selected_relevant, "fd_selected", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(xphys_values_relevant, "xphys", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(xphys_filter_relevant, "xphys_filter", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(xphys_heaviside_relevant, "xphys_heaviside", DataOut<dim>::type_dof_data);

        data_out.add_data_vector(fd_obj_relevant, "fd_obj", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[0], "adj_obj", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[3], "abs_error_obj", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[6], "rel_error_obj", DataOut<dim>::type_dof_data);

        data_out.add_data_vector(fd_flow_relevant, "fd_flow", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[1], "adj_flow", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[4], "abs_error_flow", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[7], "rel_error_flow", DataOut<dim>::type_dof_data);

        data_out.add_data_vector(fd_vol_relevant, "fd_volume", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[2], "adj_volume", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[5], "abs_error_volume", DataOut<dim>::type_dof_data);
        data_out.add_data_vector(output_relevant_vectors[8], "rel_error_volume", DataOut<dim>::type_dof_data);

        data_out.build_patches(1);

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(1)
            << perturbation;
        const std::string perturbation_str = oss.str();

        const std::string filename =
            "fd_sensitivity_loop_" +
            Utilities::int_to_string(loop_num, 4) +
            "_h_" + perturbation_str + ".vtu";

        data_out.write_vtu_in_parallel(filename, mpi_communicator);

        pcout << "Finite-difference VTU written to " << filename << std::endl;
    }
}
