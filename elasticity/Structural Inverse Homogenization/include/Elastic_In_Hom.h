#pragma once
#include <stdlib.h>
#include <fstream>
#include <iostream>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/distributed/tria.h>
// 使用Trilions
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/multigrid/mg_coarse.h>
#include "Deal_II_MMA.h"
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/base/symmetric_tensor.h>

using namespace dealii;
const int dim = 2;
const unsigned int degree = 1;

// 根据lambda和mu计算弹性张量矩阵
Tensor<2, 3> get_stress_strain_tensor(const double lambda,
                                      const double mu);
// 应变张量\epsilon = \frac{1}{2}(\nabla u + (\nabla u)^T)
// 此处计算的是对应的向量
inline Tensor<1, 3> get_strain(const FEValues<dim> &fe_values,
                                          const unsigned int   shape_func,
                                          const unsigned int   q_point)
{
  Tensor<1, 3> strain;
  for (unsigned int i = 0; i < dim; ++i)
    strain[i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

  strain[2] = (fe_values.shape_grad_component(shape_func, q_point, 0)[1] +
               fe_values.shape_grad_component(shape_func, q_point, 1)[0]);
  return strain;
}

inline Tensor<1, 3> get_strain(const std::vector<Tensor<1, dim>> &grad)
{
  Tensor<1, 3> strain;
  strain[0] = grad[0][0];
  strain[1] = grad[1][1];
  strain[2] = (grad[0][1] + grad[1][0]);

  return strain;
}
// 变换结构体
struct GridFunc
{
  // x(\zeta,\eta,\xi,\yi),x=sum(Ni*xi)
  double zeta_eta_trans_to_x(const double zeta,const double eta) const
  {
    Vector<double> N_i(4);
    N_i[0] = 0.25 * (1 + zeta) * (1 + eta);
    N_i[1] = 0.25 * (1 - zeta) * (1 + eta);
    N_i[2] = 0.25 * (1 - zeta) * (1 - eta);
    N_i[3] = 0.25 * (1 + zeta) * (1 - eta);
    return N_i[0] * edge_points[0][0] +
           N_i[1] * edge_points[1][0] +
           N_i[2] * edge_points[2][0] + 
           N_i[3] * edge_points[3][0];
  }
  // y(\zeta,\eta,\xi,\yi),y=sum(Ni*yi)
  double zeta_eta_trans_to_y(const double zeta,const double eta) const
  {
    Vector<double> N_i(4);
    N_i[0] = 0.25 * (1 + zeta) * (1 + eta);
    N_i[1] = 0.25 * (1 - zeta) * (1 + eta);
    N_i[2] = 0.25 * (1 - zeta) * (1 - eta);
    N_i[3] = 0.25 * (1 + zeta) * (1 - eta);
    return N_i[0] * edge_points[0][1] +
           N_i[1] * edge_points[1][1] +
           N_i[2] * edge_points[2][1] + 
           N_i[3] * edge_points[3][1];
  }
  // 
  Point<dim> operator()(const Point<dim> &in) const
  {
    return {
           zeta_eta_trans_to_x(in[0],in[1]),
           zeta_eta_trans_to_y(in[0],in[1])
           };
  }
  // 
  Tensor<2,dim,double> Generate_Invers_Jacobian(Point<dim> p)
  {
    Tensor<2,dim,double> Jacobian,Invers_Jacobian;
    Vector<double> dN_i_dZeta(4);
    Vector<double> dN_i_dEta(4);
    // \frac{dN_i}{d\zeta}
    dN_i_dZeta[0] =  0.25 * (1 + p[1]);
    dN_i_dZeta[1] = -0.25 * (1 + p[1]);
    dN_i_dZeta[2] = -0.25 * (1 - p[1]);
    dN_i_dZeta[3] =  0.25 * (1 - p[1]);
    // \frac{dN_i}{d\eta}
    dN_i_dEta[0] =   0.25 * (1 + p[0]);
    dN_i_dEta[1] =   0.25 * (1 - p[0]);
    dN_i_dEta[2] =  -0.25 * (1 - p[0]);
    dN_i_dEta[3] =  -0.25 * (1 + p[0]);
    // 
    Jacobian[0][0] = dN_i_dZeta[0] * edge_points[0][0] + 
                     dN_i_dZeta[1] * edge_points[1][0] + 
                     dN_i_dZeta[2] * edge_points[2][0] + 
                     dN_i_dZeta[3] * edge_points[3][0];
    // 
    Jacobian[0][1] = dN_i_dEta[0] * edge_points[0][0] + 
                     dN_i_dEta[1] * edge_points[1][0] + 
                     dN_i_dEta[2] * edge_points[2][0] + 
                     dN_i_dEta[3] * edge_points[3][0];
    // 
    Jacobian[1][0] = dN_i_dZeta[0] * edge_points[0][1] + 
                     dN_i_dZeta[1] * edge_points[1][1] + 
                     dN_i_dZeta[2] * edge_points[2][1] + 
                     dN_i_dZeta[3] * edge_points[3][1];
    // 
    Jacobian[1][1] = dN_i_dEta[0] * edge_points[0][1] + 
                     dN_i_dEta[1] * edge_points[1][1] + 
                     dN_i_dEta[2] * edge_points[2][1] + 
                     dN_i_dEta[3] * edge_points[3][1];
    // 
    Invers_Jacobian[0][0] = Jacobian[1][1];
    Invers_Jacobian[1][1] = Jacobian[0][0];
    Invers_Jacobian[1][0] = -Jacobian[1][0];
    Invers_Jacobian[0][1] = -Jacobian[0][1];
    Invers_Jacobian.operator*=(1/(Jacobian[0][0]*Jacobian[1][1] - Jacobian[0][1]*Jacobian[1][0]));
    return Invers_Jacobian;
  }  
  // 计算3x3的矩阵 
  Tensor<2,3,double> Generate_Invers_Trans(Point<dim> p)
  {
    Tensor<2,dim,double> Jacobian = Generate_Invers_Jacobian(p);
    Tensor<2,3,double> Trans;
    // 
    Trans[0][0] = Jacobian[0][0] * Jacobian[0][0];
    Trans[0][1] = Jacobian[0][1] * Jacobian[0][1];
    Trans[0][2] = Jacobian[0][0] * Jacobian[0][1];
    // 
    Trans[1][0] = Jacobian[1][0] * Jacobian[1][0];
    Trans[1][1] = Jacobian[1][1] * Jacobian[1][1];
    Trans[1][2] = Jacobian[1][0] * Jacobian[1][1];
    // 
    Trans[2][0] = 2 * Jacobian[0][0] * Jacobian[1][0];
    Trans[2][1] = 2 * Jacobian[0][1] * Jacobian[1][1];
    Trans[2][2] = Jacobian[0][0] * Jacobian[1][1] + 
                  Jacobian[0][1] * Jacobian[1][0];
    return Trans;
  }
  // 计算3x3的矩阵 
  Tensor<2,3,double> Generate_Invers_Trans_T(Point<dim> p)
  {
    Tensor<2,dim,double> Jacobian = Generate_Invers_Jacobian(p);
    Tensor<2,3,double> Trans;
    // 
    Trans[0][0] = Jacobian[0][0] * Jacobian[0][0];
    Trans[1][0] = Jacobian[0][1] * Jacobian[0][1];
    Trans[2][0] = Jacobian[0][0] * Jacobian[0][1];
    // 
    Trans[0][1] = Jacobian[1][0] * Jacobian[1][0];
    Trans[1][1] = Jacobian[1][1] * Jacobian[1][1];
    Trans[2][1] = Jacobian[1][0] * Jacobian[1][1];
    // 
    Trans[0][2] = 2 * Jacobian[0][0] * Jacobian[1][0];
    Trans[1][2] = 2 * Jacobian[0][1] * Jacobian[1][1];
    Trans[2][2] = Jacobian[0][0] * Jacobian[1][1] + 
                  Jacobian[0][1] * Jacobian[1][0];
    return Trans;
  }
  // 
  std::vector<Point<dim>> edge_points;
};
// 弹性张量均质化
class ElasticHomogenization
{
public:
  ElasticHomogenization();
  // 读取外部输入信息
  void Run();

private:
  // 划分\导入网格
  void Make_Grid();
  // 设定Trans矩阵
  void SetUpTrans();
  // 初始化Elastic系统
  void Init_Elastic_System();
  // 初始化密度模型和Simp系统
  void Init_Rho_System();
  //
  void Assemble_Elastic_System();
  void Solve_Elastic_System();
  void Generate_Homogenization();
  void Generate_Object_Diff();
  // 
  void Init_Filter_System();
  void Assemble_Filter_System();
  void Assemble_Filter_Rhs(LA::MPI::Vector Vec);
  void Solve_Filter();
  void Generate_Average_Vector(LA::MPI::Vector &out_average);
  //
  void Generate_Cell_Volume();
  void SetUp_Init_Value();
  //
  void Output_Results();
  //
  MPI_Comm mpi_communicator;
  const unsigned int this_mpi_process;
  // 网格
  parallel::distributed::Triangulation<dim> triangulation;
  // 单元类型
  FESystem<dim> fe;
  FESystem<dim> fe_filter;
  FESystem<dim> fe_simp;
  // 自由度
  DoFHandler<dim> dof_handler;
  DoFHandler<dim> dof_handler_filter;
  DoFHandler<dim> dof_handler_simp;
  // 位移场自由度索引
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  // 过滤场自由度
  IndexSet locally_owned_dofs_filter;
  IndexSet locally_relevant_dofs_filter; 
  // 密度场自由度
  IndexSet locally_owned_dofs_simp;
  // 约束
  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints_filter;
  // 输出和打印时间
  ConditionalOStream pcout;
  // 拉普拉斯矩阵
  LA::MPI::SparseMatrix elastic_matrix;
  // 整个结构的测试应变形成的右手项
  std::vector<LA::MPI::Vector> unit_test_rhs;
  // 整个结构的测试应变求解得到的位移分布
  std::vector<LA::MPI::Vector> unit_test_elastic;
  // 完全分离的解向量
  LA::MPI::Vector completely_distributed_solution;
  // 优化出来的弹性张量
  Tensor<2, 3, double> Opt_Elastic_Conductivity_Matrix;
  // PDE过滤
  LA::MPI::SparseMatrix        system_matrix_filter;
  LA::MPI::Vector              locally_relevant_solution_filter;
  LA::MPI::Vector              system_rhs_filter;
  LA::MPI::Vector              completely_distributed_solution_filter;
  // 密度插值
  // Simp法的设计变量
  LA::MPI::Vector Simp_rho;
  // Simp法过滤后的设计变量
  LA::MPI::Vector              Simp_rho_Filter;
  // Simp法设计变量的上限
  LA::MPI::Vector              Simp_rho_Max;
  // Simp法设计变量的下限
  LA::MPI::Vector              Simp_rho_Min;
  // 每个单元的测度,2D面积,3D体积
  LA::MPI::Vector Cell_Volume;
  // 优化需要的敏度和约束函数的敏度
  LA::MPI::Vector Object_Function_Diff_Values;
  std::vector<LA::MPI::Vector> Constriant_Function_Diff_Values;
  // MMA
  Deal_II_MMA_Solver                         MMA_Solver;
  // Jacbian变换矩阵
  GridFunc                      Trans_Func;
  // 材料的杨氏模量和泊松比
  double E = 1;
  double Emin = 1e-9;
  double possion = 0.3;
  // 不同维度下的测试应变场的个数
  unsigned int dimepsilon = 3;
  // 惩罚系数
  unsigned int penal = 3;
  // 体积分数
  double volfrac = 0.5;
  // 细化次数
  unsigned int init_refine_time = 7;
};