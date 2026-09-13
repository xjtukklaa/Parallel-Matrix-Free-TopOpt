#ifndef __MMA__
#define __MMA__

#include <petsc.h>

/* -----------------------------------------------------------------------------
Authors: Niels Aage
 Copyright (C) 2013-2020,
This MMA implementation is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
This Module is distributed in the hope that it will be useful,implementation
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public
License along with this Module; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
-------------------------------------------------------------------------- */
class MMA {
  public:

    MMA(PetscInt n, PetscInt m, Vec x);

    MMA(PetscInt n, PetscInt m, Vec x, PetscScalar* a, PetscScalar* c, PetscScalar* d);

    MMA(PetscInt n, PetscInt m, PetscInt itr, Vec xo1, Vec xo2, Vec U, Vec L);

    MMA(PetscInt n, PetscInt m, PetscInt itr, Vec xo1, Vec xo2, Vec U, Vec L, PetscScalar* a, PetscScalar* c,
        PetscScalar* d);

    ~MMA();


    PetscErrorCode Update(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax);


    PetscErrorCode Restart(Vec xo1, Vec xo2, Vec U, Vec L);


    PetscErrorCode SetAsymptotes(PetscScalar init, PetscScalar decrease, PetscScalar increase);




    PetscErrorCode ConstraintModification(PetscBool conMod) {
        constraintModification = conMod;
        return 0;
    };





    PetscErrorCode SetRobustAsymptotesType(PetscInt val);



    PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax, PetscScalar movelim, Vec x, Vec xmin,
                                     Vec xmax);


    PetscErrorCode KKTresidual(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax, PetscScalar* norm2,
                               PetscScalar* normInf);



    PetscScalar DesignChange(Vec x, Vec xold);

  private:

    PetscErrorCode GenSub(Vec xval, Vec dfdx, PetscScalar* gx, Vec* dgdx, Vec xmin, Vec xmax);


    PetscErrorCode SolveDIP(Vec xval);


    PetscErrorCode XYZofLAMBDA(Vec x);


    PetscErrorCode DualGrad(Vec x);


    PetscErrorCode DualHess(Vec x);


    PetscErrorCode DualLineSearch();


    PetscScalar DualResidual(Vec x, PetscScalar epsi);


    PetscInt n, m, k;


    PetscScalar asyminit, asymdec, asyminc;


    PetscBool constraintModification;


    PetscBool NonLinConstraints;



    PetscInt RobustAsymptotesType;


    PetscScalar *a, *c, *d;


    PetscScalar* y;
    PetscScalar  z;


    PetscScalar *lam, *mu, *s;


    Vec L, U, alpha, beta, p0, q0, *pij, *qij;


    PetscScalar *b, *grad, *Hess;


    Vec xo1, xo2;


    PetscErrorCode Factorize(PetscScalar* K, PetscInt nn);
    PetscErrorCode Solve(PetscScalar* K, PetscScalar* x, PetscInt nn);
    PetscScalar    Min(PetscScalar d1, PetscScalar d2);
    PetscScalar    Max(PetscScalar d1, PetscScalar d2);
    PetscInt       Min(PetscInt d1, PetscInt d2);
    PetscInt       Max(PetscInt d1, PetscInt d2);
    PetscScalar    Abs(PetscScalar d1);
};

#endif
