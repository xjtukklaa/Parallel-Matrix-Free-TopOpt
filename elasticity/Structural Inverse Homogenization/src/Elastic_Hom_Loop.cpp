#include "../include/Elastic_In_Hom.h"

ElasticHomogenization::ElasticHomogenization()
    : mpi_communicator(MPI_COMM_WORLD),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening)),
      fe(FE_Q<dim>(degree)^dim),
      fe_filter(FE_Q<dim>(degree)),
      fe_simp(FE_DGQ<dim>(0)),
      dof_handler(triangulation),
      dof_handler_filter(triangulation),
      dof_handler_simp(triangulation),
      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      MMA_Solver()
{}

Tensor<2, 3> get_stress_strain_tensor(const double lambda,
                                      const double mu)
{
  Tensor<2, 3> stress_strain_tensor;
  stress_strain_tensor = 0;
  stress_strain_tensor[0][0] = lambda + 2 * mu;
  stress_strain_tensor[0][1] = lambda;
  stress_strain_tensor[1][0] = lambda;
  stress_strain_tensor[1][1] = lambda + 2 * mu;
  stress_strain_tensor[2][2] = mu;
  return stress_strain_tensor;
}

void ElasticHomogenization::SetUp_Init_Value()
{
  std::vector<types::global_dof_index> simp_dof_indices(1);
  for (auto cell_iter : dof_handler_simp.active_cell_iterators())
  {
    if (cell_iter->is_locally_owned())
    {
      cell_iter->get_dof_indices(simp_dof_indices);
      // 初始构型
      if (std::pow(cell_iter->center()(0),2) + 
          std::pow(cell_iter->center()(1),2) < std::pow(0.5,2))
      {
        Simp_rho[simp_dof_indices[0]] = 0.9;
      }else{
        Simp_rho[simp_dof_indices[0]] = 1.;
      }                   
    }
  }
  Simp_rho.compress(VectorOperation::insert);
}

void ElasticHomogenization::SetUpTrans()
{
  Trans_Func.edge_points.resize(4);
  // (Point,Dim)
  // 左下角点开始,逆时针旋转
  Trans_Func.edge_points[0][0] = -1;
  Trans_Func.edge_points[0][1] = -1;
  Trans_Func.edge_points[1][0] =  1;
  Trans_Func.edge_points[1][1] = -1;
  Trans_Func.edge_points[2][0] =  1;
  Trans_Func.edge_points[2][1] =  1;
  Trans_Func.edge_points[3][0] = -1;
  Trans_Func.edge_points[3][1] =  1;
}

void ElasticHomogenization::Run()
{
  // 导入网格
  Make_Grid();
  // 设定TransFunc
  SetUpTrans();
  // 初始化系统
  Init_Rho_System();   
  Init_Filter_System();
  Init_Elastic_System();
  Generate_Cell_Volume();
  // 初始化MMA
  MMA_Solver.Deal_II_MMA_Init(1, 1., 1000., Simp_rho);
  // 初始化MMA的迭代数
  MMA_Solver.Loop_Iter = 0;
  pcout << "Number Of Cells : " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number Of Dofs : " << dof_handler.n_dofs() << std::endl;
  // 初始化体积约束的敏度
  Assemble_Filter_System(); 
  Assemble_Filter_Rhs(Cell_Volume);
  Solve_Filter();
  Generate_Average_Vector(Constriant_Function_Diff_Values[0]);
  // 体积的最大值
  double const_max = Constriant_Function_Diff_Values[0].linfty_norm();
  Constriant_Function_Diff_Values[0] /= const_max;
  // rho的变化量
  double change = 1e10;
  // 用于限制MMA的移动
  double move = 0.1;
  // 约束函数的值
  Vector<double> Constriant_Function_Value(1);
  // 拓扑优化的主循环
  while (MMA_Solver.Loop_Iter < 35 && change > 1e-3 * dof_handler_simp.n_dofs())
  {
    MMA_Solver.Loop_Iter ++;
    if (MMA_Solver.Loop_Iter == 1)
    {
      SetUp_Init_Value();
    }
    // 
    for (unsigned int i : locally_owned_dofs_simp)
    {
      Simp_rho_Max[i] = std::min(Simp_rho[i] + move, 1.);
      Simp_rho_Min[i] = std::max(Simp_rho[i] - move, 0.);
    }      
    Simp_rho_Max.compress(VectorOperation::insert);
    Simp_rho_Min.compress(VectorOperation::insert);
    // 
    Assemble_Filter_Rhs(Simp_rho);
    Solve_Filter();
    Generate_Average_Vector(Simp_rho_Filter);
    // 
    Assemble_Elastic_System();
    Solve_Elastic_System();
    Generate_Homogenization();
    // 
    pcout<<"The Hom Tensor : "<<std::endl;
    for (unsigned int i = 0; i < dimepsilon; i++)
    {
      for (unsigned int j = 0; j < dimepsilon; j++)
      {
        pcout<<Opt_Elastic_Conductivity_Matrix[i][j]<<"\t";
      }
      pcout<<std::endl;
    }    
    pcout<<std::endl;
    Generate_Object_Diff();
    // 
    Assemble_Filter_Rhs(Object_Function_Diff_Values);
    Solve_Filter();
    Generate_Average_Vector(Object_Function_Diff_Values);
    Object_Function_Diff_Values /= Object_Function_Diff_Values.linfty_norm();
    // 
    Constriant_Function_Value[0] = Simp_rho * Cell_Volume;
    Constriant_Function_Value[0] -= volfrac * Cell_Volume.l1_norm();
    Constriant_Function_Value[0] /= const_max;
    // MMA
    MMA_Solver.Deal_II_MMA_Dual_Problem_Solve(Simp_rho,
                                              Object_Function_Diff_Values,
                                              Constriant_Function_Value,
                                              Constriant_Function_Diff_Values,
                                              Simp_rho_Max,
                                              Simp_rho_Min);
    change = MMA_Solver.Deal_II_MMA_Get_Change(Simp_rho);
    pcout<< "It : "<<MMA_Solver.Loop_Iter
         << " Obj : "<<Opt_Elastic_Conductivity_Matrix[0][1] -
                        std::pow(0.8,(double)MMA_Solver.Loop_Iter) *
                        (Opt_Elastic_Conductivity_Matrix[0][0] + 
                          Opt_Elastic_Conductivity_Matrix[1][1])
          <<" Change : "<<change
          <<" Vol : "<<(Simp_rho * Cell_Volume) / Cell_Volume.l1_norm()<<std::endl; 
  }
  Output_Results();  
}