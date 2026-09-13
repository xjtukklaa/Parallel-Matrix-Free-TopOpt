#pragma once
#include <deal.II/lac/generic_linear_algebra.h>
using namespace dealii;
// #define FORCE_USE_OF_TRILINOS
// PETSC或者Trilions的线性代数库
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
!(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
    using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
    using namespace dealii::LinearAlgebraTrilinos;
#  define USE_TRILINOS_LA
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

class Deal_II_MMA_Solver
{
public:
	Deal_II_MMA_Solver();
	// 初始化MMA
	void Deal_II_MMA_Init(unsigned int Constraints,
						  double A_i,
						  double C_i,
						  LA::MPI::Vector MMA_Design_Variables);
	// 内点法求解
	void Deal_II_MMA_Dual_Problem_Solve(LA::MPI::Vector &MMA_Design_Variables,
										LA::MPI::Vector MMA_Object_Function_Diff_Values,
										Vector<double>  MMA_Constraints_Function_Values,
										std::vector<LA::MPI::Vector> MMA_Constraints_Function_Diff_Values,
										LA::MPI::Vector MMA_Xmax,
										LA::MPI::Vector MMA_Xmin);
	// 网格变动后重新初始化
	void Deal_II_MMA_Reinit(LA::MPI::Vector MMA_Design_Variables);
	// 设定变动参数
	void Deal_II_MMA_SetAsymptotes(double init, double decrease, double increase);
	// 设定是否使用更加鲁棒的MMA
	void Deal_II_MMA_SetRobustAsymptotesType(bool robust_type);
	// 设定是否将约束函数凸近似化
	void Deal_II_MMA_SetConstraintConvexApproximationType(bool constraint_type);
	// 获得当前向量和上一次的变动量
	double Deal_II_MMA_Get_Change(LA::MPI::Vector MMA_Design_Variables);
	// 计算KKT残差的二范数
	double Deal_II_MMA_KKT_L2_Norm(LA::MPI::Vector MMA_Design_Variables,
								   LA::MPI::Vector MMA_Object_Function_Diff_Values,
								   Vector<double>  MMA_Constraints_Function_Values,
								   std::vector<LA::MPI::Vector> MMA_Constraints_Function_Diff_Values,
								   LA::MPI::Vector MMA_Xmax,
								   LA::MPI::Vector MMA_Xmin);
	// 当前的循环次数，由外部输入，不在内部进行记录
	int Loop_Iter;
	// MMA优化时的移动上限和下限
	// xmin<Low<beta<x<alpha<Upp<xmax
	LA::MPI::Vector Moving_Low_Boundary, Moving_Upp_Boundary;
	// MMA储存的前两次的设计变量的信息
	std::vector<LA::MPI::Vector> Old_Design_Variables;
private:
	// 根据输入的目标函数的导数和约束函数的导数
	// 建立近似子问题
	void Generate_SubConvexProblem(LA::MPI::Vector MMA_Design_Variables,
								   LA::MPI::Vector MMA_Object_Function_Diff_Values,
								   Vector<double> MMA_Constraints_Function_Values,
								   std::vector<LA::MPI::Vector> MMA_Constraints_Function_Diff_Values,
								   LA::MPI::Vector MMA_Xmax,
								   LA::MPI::Vector MMA_Xmin);
	// 内点法求解器
	void Interior_Point_Solve(LA::MPI::Vector &MMA_Design_Variables);
	// 获取对偶梯度和hess矩阵
	void Get_Dual_Gradient(LA::MPI::Vector MMA_Design_Variables);
	void Get_Dual_Hess(LA::MPI::Vector MMA_Design_Variables);
	// 根据对偶变量Lambda计算得到X,Y,Z
	void Generate_XYZ_From_Lambda(LA::MPI::Vector &MMA_Design_Variables);
	// 在得到搜索方向后，回溯线性搜索步长
	void Dual_Linear_Search();
	// 求解对偶线性系统
	void Solve_Linear_System();
	// 计算对偶误差
	double Get_Dual_Residual(LA::MPI::Vector MMA_Design_Variables, double Epsi);
	// 约束个数
	unsigned int Number_Of_Constraints;
	// 原始对偶残差
	double Primal_Dual_Epsi;
	// 在迭代过程中的xmax上限和xmin下限的最小差值
	double Xmax_Xmin_Epsi;
	// 近似的凸问题的控制参数
	// 用于调节近似子问题与原始问题的相似程度
	// 在MMA中不变，GCMMA中会变化
	double Rho_Epsi;
	// 计算alpha上限和beta下限的参数
	double Alpha_Beta;
	// 用于控制MMA迭代时的约束放松和收缩的参数
	// 主要影响优化次数和优化的稳健性
	double Asym_Init, Asym_Decrease, Asym_Increase;
	// 用于循环的MMA自由度索引集合
	IndexSet MMA_Dofs_IndexSet;
	// 子问题构建参数
	// 在这个代码里面目标函数构建子问题的
	// 参数a0和d0都是1
	Vector<double> SubProblem_A, SubProblem_B, SubProblem_C;
	// 子问题变量
	Vector<double> SubProblem_Y;
	double SubProblem_Z;

	// 对偶变量
	Vector<double> DualProblem_Lambda, DualProblem_Mu;
	// 为了将不等式约束条件转化为等式条件
	// 引入的松弛变量
	Vector<double> Slack_Variables;

	// MMA优化时的移动上限和下限
	// xmin<Low<beta<x<alpha<Upp<xmax
	LA::MPI::Vector Moving_Alpha_Boundary, Moving_Beta_Boundary;

	// MMA构建的目标函数的梯度信息
	LA::MPI::Vector Object_Function_Positive, Object_Function_Negative;
	std::vector<LA::MPI::Vector> Constraint_Function_Positive, Constraint_Function_Negative;

	// 内点法求解时需要的梯度和海斯矩阵
	Vector<double> Object_Function_Gradient;
	FullMatrix<double> Object_Function_Hess;
	// 判断是否将约束函数也开启凸近似
	bool Constraint_Convex_Approximation;
	// 判断MMA是否严格鲁棒
	bool RobustAsymptotesType;
};