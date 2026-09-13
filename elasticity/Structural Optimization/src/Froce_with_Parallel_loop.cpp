#include "../include/Force_with_Parallel.h"
#include <iomanip>
ForceProblem::ForceProblem()
	: mpi_communicator(MPI_COMM_WORLD),
	  this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
	  triangulation(mpi_communicator,
					typename Triangulation<dim>::MeshSmoothing(
						Triangulation<dim>::smoothing_on_refinement |
						Triangulation<dim>::smoothing_on_coarsening)),
	  mapping_fe(MappingFE<dim>(FE_Q<dim>(degree))),
	  mapping_fe_filter(MappingFE<dim>(FE_Q<dim>(degree))),
	  fe(FESystem<dim>(FE_Q<dim>(degree)^dim), FESystem<dim>(FE_Nothing<dim>()^dim)),
	  fe_rho(FE_DGQ<dim>(0), FE_Nothing<dim>()),	
	  fe_filter(FE_Q<dim>(degree), FE_Nothing<dim>()),	    
	  quadrature_formula(QGauss<dim>(degree + 1)),
	  quadrature_formula_face(QGauss<dim - 1>(degree + 1)),
	  quadrature_formula_rho(QGauss<dim>(1)),
	  quadrature_formula_filter(QGauss<dim>(degree+1)),
	  dof_handler(triangulation),
	  dof_handler_rho(triangulation),
	  dof_handler_filter(triangulation),
	  pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
	  computing_timer(mpi_communicator,
					  pcout,
					  TimerOutput::never,
					  TimerOutput::wall_times),
	  MMA_Solver()
{}

void ForceProblem::init_simp()
{
	TimerOutput::Scope t(computing_timer, "init_simp");
	// 密度场
	for (const auto &cell : dof_handler_rho.active_cell_iterators())
	{
		if (cell->is_locally_owned())
		{
			cell->set_active_fe_index(0);		
		}
	}
	// 
	dof_handler_rho.distribute_dofs(fe_rho);
	locally_owned_dofs_rho = dof_handler_rho.locally_owned_dofs();
	Simp_Rho.reinit(locally_owned_dofs_rho, mpi_communicator);
	Simp_Rho_Filted.reinit(locally_owned_dofs_rho, mpi_communicator);
	Simp_Max.reinit(locally_owned_dofs_rho, mpi_communicator);
	Simp_Min.reinit(locally_owned_dofs_rho, mpi_communicator);
	Object_Diff_Values.reinit(locally_owned_dofs_rho, mpi_communicator);
	Constraint_Diff_Values.resize(1);
	Constraint_Diff_Values[0].reinit(locally_owned_dofs_rho, mpi_communicator);
	Cell_Volume.reinit(locally_owned_dofs_rho, mpi_communicator);
}

void ForceProblem::run()
{
	make_grid();
	// 
	setup_system();
	setup_filter_system();
	init_simp();
	// 
	pcout<<"Number Of Dofs : "<<dof_handler.n_dofs()<<std::endl
		<<"Number Of Cells : "<<triangulation.n_active_cells()<<std::endl;

	double change = 1e10;
	double vol_now = 0;
	double Object_Funtion_Values = 0;
	Vector<double> Constraint_Function_Values(1);
	MMA_Solver.Deal_II_MMA_Init(1,0.,1e3,Simp_Rho);
	MMA_Solver.Loop_Iter = 0;
	while(MMA_Solver.Loop_Iter < Loop_Max && change > 1e-4 * triangulation.n_active_cells())
	{
		change = 0;
		MMA_Solver.Loop_Iter ++;
		Object_Diff_Values[0] = 0;
		Constraint_Function_Values[0] = 0;
		if (MMA_Solver.Loop_Iter == 1)
		{
			Simp_Max = 1.;
			Simp_Min = 0.;
			Simp_Rho = 1.;
			// 
			get_cell_volume();
			// 
			assemble_filter_system();
			// 
			assemble_filter_rhs(Cell_Volume);
			solve_filter();
			generate_average_vector(Constraint_Diff_Values[0]);
		}
		// 
		assemble_filter_rhs(Simp_Rho);
		solve_filter();
		generate_average_vector(Simp_Rho_Filted);
		Simp_Rho_Filted = Simp_Rho;
		// 
		assemble_system();
		assemble_system_rhs();
		solve();
		get_object_diff_values(Object_Diff_Values);
		// 
		assemble_filter_rhs(Object_Diff_Values);
		solve_filter();
		generate_average_vector(Object_Diff_Values);
	    // 
		vol_now = Cell_Volume * Simp_Rho_Filted;
        Constraint_Function_Values[0] = (vol_now - (Cell_Volume.l1_norm() * volfrac));
		//
		Object_Funtion_Values = system_rhs * completely_distributed_solution;
		// 
		MMA_Solver.Deal_II_MMA_Dual_Problem_Solve(Simp_Rho,
		                                          Object_Diff_Values,
												  Constraint_Function_Values,
												  Constraint_Diff_Values,
												  Simp_Max,
												  Simp_Min);
		change = MMA_Solver.Deal_II_MMA_Get_Change(Simp_Rho);
		// pcout<< std::fixed << std::setprecision(5)
		//      <<"It : "<<MMA_Solver.Loop_Iter<<"\t"
		// 	 <<" Change : "<<change<<"\t"
		// 	 <<" Object : "<<Object_Funtion_Values<<"\t"
		// 	 <<" Const : "<<Constraint_Function_Values[0]<<"\t"
		// 	 <<" Vol : "<<vol_now/Cell_Volume.l1_norm()<<"\t"
		// 	 <<std::endl;		
		// if (MMA_Solver.Loop_Iter % 15 == 0)
		// {
		// 	Refine_Grid();
		// 	// 
		// 	get_cell_volume();
		// 	// 
		// 	assemble_filter_system();
		// 	// 
		// 	assemble_filter_rhs(Cell_Volume);
		// 	solve_filter();
		// 	generate_average_vector(Constraint_Diff_Values[0]);
		// }
		output_results(MMA_Solver.Loop_Iter);
	}
}

void ForceProblem::time()
{
	computing_timer.print_summary();
	computing_timer.reset();
	pcout << std::endl;
}