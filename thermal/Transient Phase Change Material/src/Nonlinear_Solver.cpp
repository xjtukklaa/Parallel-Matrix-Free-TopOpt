#include "../include/Nonlinear_Time.h"

Nonlinear_Time::Nonlinear_Time() 
:   mpi_communicator(MPI_COMM_WORLD),
    triangulation(mpi_communicator,
        typename Triangulation<dim>::MeshSmoothing(
            Triangulation<dim>::smoothing_on_refinement |
            Triangulation<dim>::smoothing_on_coarsening)),
    fe(degree),
    fe_filter(degree),
    fe_simp(0),
    dof_handler(triangulation),
    dof_handler_filter(triangulation),
    dof_handler_simp(triangulation),
    time_stepper_data("",
                      /* ts type */ "beuler",
                      /* start time */ 0.0,
                      /* end time */ 100.0,
                      /* initial time step */ 1e-4,
                      /* max steps */   -1,
                      /* match step */  true,
                      /* restart if remesh */  false,
                      /* ts adapt type */ "dsp"),
    initial_global_refinement(7),
    pcout(std::cout, 
          Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
    computing_timer(pcout, 
                    TimerOutput::summary, 
                    TimerOutput::wall_times)
{}

void Nonlinear_Time::make_grid()
{
    TimerOutput::Scope t(computing_timer, "make_grid");
    // GridGenerator::hyper_cube(triangulation, -1, 1);
    GridGenerator::hyper_ball(triangulation, Point<dim>(0.,0.), 1, true);
    for (auto & face : triangulation.active_face_iterators())
    {
        if (face->at_boundary())
        {
            face->set_boundary_id(1);
        }
    }    
    triangulation.refine_global(initial_global_refinement);
    // GridIn<dim> grid_in(triangulation);
    // grid_in.read_msh("input.msh");
    // triangulation.refine_global(1);
}

void Nonlinear_Time::run()
{
    make_grid();

    setup_forward_system();
    setup_adjoint_system();
    setup_filter_system();

    MMA_Solver.Loop_Iter = 0;
    MMA_Solver.Deal_II_MMA_SetRobustAsymptotesType(true);
    MMA_Solver.Deal_II_MMA_SetConstraintConvexApproximationType(false);
    MMA_Solver.Deal_II_MMA_Init(1, 0., 1.e3, simp_rho);
    
    get_cell_volume();

    Rmin = 2 * std::pow(cell_volume.linfty_norm(),1./(double)dim) / (2.* std::sqrt(3));
    assemble_filter_matrix();

    Vector<double> Object_Value(1);

    double change = 1e10;
    double volume_now = 0;
    double const_max = 0;
    Vector<double> Constraint_Value(1);

    helmholtz_filter(cell_volume, Constraint_Function_Derivative[0]);
    const_max = Constraint_Function_Derivative[0].linfty_norm();
    Constraint_Function_Derivative[0] /= const_max;

    while (MMA_Solver.Loop_Iter < 1000 && change > 1e-4 * dof_handler_simp.n_dofs())
    {
        MMA_Solver.Loop_Iter ++;

        if (MMA_Solver.Loop_Iter == 1)
        {
            simp_rho_max = 1.;
            simp_rho_min = 1e-3;
            simp_rho = 1.;            
        }

        // VectorTools::interpolate(dof_handler,
        //                          Functions::ConstantFunction<dim>(293.15),
        //                          tempture_solution);
        tempture_solution = 293.15;
        
        helmholtz_filter(simp_rho, simp_rho_filter);

        setup_forward_solver();
        setup_adjoint_solver();

        pcout<< "Forward Time Solve "<< std::endl;
        time_stepper.solve(tempture_solution);

        Object_Value[0] =  tempture_solution.l1_norm()/dof_handler.n_dofs();
        
        initialize_adjoint_vector();
        TSSetCostGradients(time_stepper.petsc_ts(),
                            1, 
                            &adjoint_lambda.petsc_vector(), 
                            &adjoint_mu.petsc_vector());

        pcout<< "Adjoint Time Solve "<< std::endl;
        TSAdjointSolve(time_stepper.petsc_ts());

        helmholtz_filter(adjoint_mu, Object_Function_Derivative);
        Object_Function_Derivative /= Object_Function_Derivative.linfty_norm();

        volume_now = simp_rho * cell_volume;
        Constraint_Value[0] = volume_now - cell_volume.l1_norm() * volfrac;
        Constraint_Value[0] /= const_max;

        MMA_Solver.Deal_II_MMA_Dual_Problem_Solve(simp_rho,
                                                  Object_Function_Derivative,
                                                  Constraint_Value,
                                                  Constraint_Function_Derivative,
                                                  simp_rho_max,
                                                  simp_rho_min);

        change = MMA_Solver.Deal_II_MMA_Get_Change(simp_rho);

        pcout<<"It : "<<MMA_Solver.Loop_Iter<<"\t"
             <<" Change : "<<change<<"\t"
             <<" Object : "<<Object_Value[0]<<"\t"
             <<" vol : "<<volume_now/cell_volume.l1_norm()<<"\t"
             <<std::endl;

        output_sensitivity_optimization(MMA_Solver.Loop_Iter);

        TSReset(quadts);
        TSTrajectoryReset(tj);        
        // time();
    }
}

void Nonlinear_Time::output_sensitivity_optimization(unsigned int iter)
{
    TimerOutput::Scope t(computing_timer, "output_sensitivity_optimization");
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_simp);    
    data_out.add_data_vector(dof_handler_simp, simp_rho, "simp_rho");
    data_out.add_data_vector(dof_handler_simp, simp_rho_filter, "simp_rho_filter");
    data_out.add_data_vector(dof_handler_simp, simp_rho_max, "simp_rho_max");
    data_out.add_data_vector(dof_handler_simp, simp_rho_min, "simp_rho_min");

    data_out.add_data_vector(dof_handler_simp, adjoint_mu, "adjoint_mu");
    data_out.add_data_vector(dof_handler_simp, Object_Function_Derivative, "Object_Function_Derivative");

    data_out.add_data_vector(dof_handler_simp, cell_volume, "cell_volume");
    data_out.add_data_vector(dof_handler_simp, Constraint_Function_Derivative[0], "Constraint_Function_Derivative");

    data_out.build_patches();
    std::string filename = "optimization_solution_"+ Utilities::to_string(iter,4) +".vtu";
    data_out.write_vtu_in_parallel(filename, mpi_communicator);
}

void Nonlinear_Time::time()
{
    computing_timer.print_summary();
    computing_timer.reset();
}