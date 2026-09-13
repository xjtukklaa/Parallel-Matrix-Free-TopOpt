#pragma once
#include <deal.II/lac/generic_linear_algebra.h>
using namespace dealii;

#if defined FORCE_USE_PETSC_LA
    using MMAVectorType = dealii::LinearAlgebraPETSc::MPI::Vector;
#elif defined FORCE_USE_TRILINOS_LA
    using MMAVectorType = dealii::LinearAlgebraTrilinos::MPI::Vector;
#else
    using MMAVectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    # define COMPRESS_ERROR
#endif

class Deal_II_MMA_Solver
{
public:
	Deal_II_MMA_Solver();
	// Initialize MMA
	void Deal_II_MMA_Init(unsigned int Constraints,
						  double A_i,
						  double C_i,
						  MMAVectorType MMA_Design_Variables);
	// Solve by interior point method
	void Deal_II_MMA_Dual_Problem_Solve(MMAVectorType &MMA_Design_Variables,
										MMAVectorType MMA_Object_Function_Diff_Values,
										Vector<double>  MMA_Constraints_Function_Values,
										std::vector<MMAVectorType> MMA_Constraints_Function_Diff_Values,
										MMAVectorType MMA_Xmax,
										MMAVectorType MMA_Xmin);
	// Reinitialize after mesh change
	void Deal_II_MMA_Reinit(MMAVectorType MMA_Design_Variables);
	// Set move parameters
	void Deal_II_MMA_SetAsymptotes(double init, double decrease, double increase);
	// Set whether to use more robust MMA
	void Deal_II_MMA_SetRobustAsymptotesType(bool robust_type);
	// Set whether to use convex approximation for constraint functions
	void Deal_II_MMA_SetConstraintConvexApproximationType(bool constraint_type);
	// Get the change between current and previous design variables
	double Deal_II_MMA_Get_Change(MMAVectorType MMA_Design_Variables);
	// Compute the L2 norm of the KKT residual
	double Deal_II_MMA_KKT_L2_Norm(MMAVectorType MMA_Design_Variables,
								   MMAVectorType MMA_Object_Function_Diff_Values,
								   Vector<double>  MMA_Constraints_Function_Values,
								   std::vector<MMAVectorType> MMA_Constraints_Function_Diff_Values,
								   MMAVectorType MMA_Xmax,
								   MMAVectorType MMA_Xmin);
	// Current iteration number, provided externally, not recorded internally
	int Loop_Iter;
	// Moving upper and lower bounds for MMA optimization
	// xmin < Low < beta < x < alpha < Upp < xmax
	MMAVectorType Moving_Low_Boundary, Moving_Upp_Boundary;
	// Storage for the design variables of the previous two iterations
	std::vector<MMAVectorType> Old_Design_Variables;
private:
	// Build the approximate subproblem based on the input derivatives
	// of the objective and constraint functions
	void Generate_SubConvexProblem(MMAVectorType MMA_Design_Variables,
								   MMAVectorType MMA_Object_Function_Diff_Values,
								   Vector<double> MMA_Constraints_Function_Values,
								   std::vector<MMAVectorType> MMA_Constraints_Function_Diff_Values,
								   MMAVectorType MMA_Xmax,
								   MMAVectorType MMA_Xmin);
	// Interior point solver
	void Interior_Point_Solve(MMAVectorType &MMA_Design_Variables);
	// Compute dual gradient and Hessian matrix
	void Get_Dual_Gradient(MMAVectorType MMA_Design_Variables);
	void Get_Dual_Hess(MMAVectorType MMA_Design_Variables);
	// Compute X, Y, Z from dual variables Lambda
	void Generate_XYZ_From_Lambda(MMAVectorType &MMA_Design_Variables);
	// Backtracking line search for step size after obtaining search direction
	void Dual_Linear_Search();
	// Solve the dual linear system
	void Solve_Linear_System();
	// Compute dual residual
	double Get_Dual_Residual(MMAVectorType MMA_Design_Variables, double Epsi);
	// Number of constraints
	unsigned int Number_Of_Constraints;
	// Primal-dual residual
	double Primal_Dual_Epsi;
	// Minimum difference between xmax upper bound and xmin lower bound during iterations
	double Xmax_Xmin_Epsi;
	// Control parameter for the approximate convex problem
	// Used to adjust the similarity between the approximate subproblem and the original problem
	// Constant in MMA, varies in GCMMA
	double Rho_Epsi;
	// Control parameter for the change of x in each iteration
	// This is the proportional change relative to xmax-xmin, not the actual magnitude
	double Design_Change_Move;
	// Parameter for computing alpha upper bound and beta lower bound
	double Alpha_Beta;
	// Parameters controlling constraint relaxation and contraction during MMA iterations
	// Mainly affects the number of optimization iterations and robustness
	double Asym_Init, Asym_Decrease, Asym_Increase;
	// Index set of MMA degrees of freedom used for looping
	IndexSet MMA_Dofs_IndexSet;
	// Subproblem construction parameters
	// In this code, the parameters a0 and d0 for constructing the subproblem
	// of the objective function are both 1
	Vector<double> SubProblem_A, SubProblem_B, SubProblem_C;
	// Subproblem variables
	Vector<double> SubProblem_Y;
	double SubProblem_Z;

	// Dual variables
	Vector<double> DualProblem_Lambda, DualProblem_Mu;
	// Slack variables introduced to convert inequality constraints
	// into equality constraints
	Vector<double> Slack_Variables;

	// Moving upper and lower bounds for MMA optimization
	// xmin < Low < beta < x < alpha < Upp < xmax
	MMAVectorType Moving_Alpha_Boundary, Moving_Beta_Boundary;

	// Gradient information of the objective function constructed by MMA
	MMAVectorType Object_Function_Positive, Object_Function_Negative;
	std::vector<MMAVectorType> Constraint_Function_Positive, Constraint_Function_Negative;

	// Gradient and Hessian matrix required for interior point method
	Vector<double> Object_Function_Gradient;
	FullMatrix<double> Object_Function_Hess;
	// Flag indicating whether to enable convex approximation for constraint functions
	bool Constraint_Convex_Approximation;
	// Flag indicating whether MMA is strictly robust
	bool RobustAsymptotesType;
};