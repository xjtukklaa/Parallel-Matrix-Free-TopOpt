#include "../../include/top_opt_heat_flow.h"

namespace TopOpt
{

    void TopOptHeatFlow::make_grid()
    {
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("../input/CAD_DesignDomain_ms070.mphtxt");
        grid_in.read_comsol_mphtxt(input_file);

        switch (PRM::dimensionless_method)
        {
        case PRM::dimensionless_on:
        {
            GridTools::scale(1.0 / PRM::L_refer, triangulation);
            break;
        }

        case PRM::dimensionless_off:
            break;

        default:
            break;
        }

        triangulation.refine_global(PRM::pre_refinement);
        define_grid();
        get_grid_information();
    }

    void TopOptHeatFlow::define_grid()
    {
        double scale_factor = 1.0;
        switch (PRM::dimensionless_method)
        {
        case PRM::dimensionless_on:
        {
            scale_factor = PRM::L_refer;
            break;
        }

        case PRM::dimensionless_off:
        {
            scale_factor = 1.0;
            break;
        }

        default:
            break;
        }
        for (const auto &cell : triangulation.cell_iterators())
        {
            if ((cell->center()(0) < 0.0) ||
                (cell->center()(0) > (10e-3 / scale_factor)))
                cell->set_material_id(1);

            else
                cell->set_material_id(0);
        }
        for (auto &face : triangulation.active_face_iterators())
        {
            if (face->at_boundary())
            {

                if (std::abs(face->center()(0) - (-1e-3 / scale_factor)) < 1e-8)
                {
                    face->set_boundary_id(1);
                }

                else if (std::abs(face->center()(0) - (11e-3 / scale_factor)) < 1e-8)
                {
                    face->set_boundary_id(2);
                }

                else if (std::abs(face->center()(1) - (5e-3 / scale_factor)) < 1e-8)
                {
                    face->set_boundary_id(3);
                }

                else
                {
                    face->set_boundary_id(0);
                }
            }
        }
    }

    void TopOptHeatFlow::get_grid_information()
    {
        TimerOutput::Scope t(computing_timer, "get_grid_information");

        /*待计算的量*/
        double volume_design = 0;
        double volume_fluid = 0;
        double volume_solid = 0;
        double area_noslip = 0;
        double area_inlet = 0;
        double area_outlet = 0;
        double area_slip = 0;
        double area_heatflux = 0;

        /*迭代器*/
        for (const auto &cell : triangulation.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                if (cell->material_id() == 0)
                    volume_design += cell->measure();
                else if (cell->material_id() == 1)
                    volume_fluid += cell->measure();
                else if (cell->material_id() == 2)
                    volume_solid += cell->measure();

                for (const auto &face : cell->face_iterators())
                {
                    if (face->at_boundary())
                    {
                        if (face->boundary_id() == 0)
                            area_noslip += face->measure();
                        else if (face->boundary_id() == 1)
                            area_inlet += face->measure();
                        else if (face->boundary_id() == 2)
                            area_outlet += face->measure();
                        else if (face->boundary_id() == 3)
                            area_slip += face->measure();
                        else if (face->boundary_id() == 4)
                            area_heatflux += face->measure();
                    }
                }
            }
        }
        {

            domain_volume[0] = Utilities::MPI::sum(volume_design, mpi_communicator);
            domain_volume[1] = Utilities::MPI::sum(volume_fluid, mpi_communicator);
            domain_volume[2] = Utilities::MPI::sum(volume_solid, mpi_communicator);

            boundary_area[0] = Utilities::MPI::sum(area_noslip, mpi_communicator);
            boundary_area[1] = Utilities::MPI::sum(area_inlet, mpi_communicator);
            boundary_area[2] = Utilities::MPI::sum(area_outlet, mpi_communicator);
            boundary_area[3] = Utilities::MPI::sum(area_slip, mpi_communicator);
            boundary_area[4] = Utilities::MPI::sum(area_heatflux, mpi_communicator);
        }

        pcout << std::endl
              << "******************************************************************************" << std::endl
              << "Present Grid information: " << std::endl
              << "Totol volume = " << domain_volume[0] + domain_volume[1] + domain_volume[2] << std::endl
              << "Totol Area = " << boundary_area[0] + boundary_area[1] + boundary_area[2] + boundary_area[3] + boundary_area[4] << std::endl
              << "Volume of domain 0 = " << domain_volume[0] << std::endl
              << "Volume of domain 1 = " << domain_volume[1] << std::endl
              << "Volume of domain 2 = " << domain_volume[2] << std::endl
              << "Area of boundary 0 = " << boundary_area[0] << std::endl
              << "Area of boundary 1 = " << boundary_area[1] << std::endl
              << "Area of boundary 2 = " << boundary_area[2] << std::endl
              << "Area of boundary 3 = " << boundary_area[3] << std::endl
              << "Area of boundary 4 = " << boundary_area[4] << std::endl;
    }

}