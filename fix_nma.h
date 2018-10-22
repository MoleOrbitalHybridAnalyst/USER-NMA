#ifdef FIX_CLASS

FixStyle(nma,FixNMA)

#else

#ifndef LMP_FIX_NMA_H
#define LMP_FIX_NMA_H

#include "fix.h"
#include "hdf5.h"
#include "EVB_api.h"

namespace LAMMPS_NS {

class FixNMA : public Fix {
 public:
   FixNMA(class LAMMPS *, int, char **);
   ~FixNMA();
   int setmask();
   void init();
   void setup(int);
   void pre_force(int);
 private:
   FixEVB* fix_evb;

   hid_t hdf5_file, dataset;
   hid_t datatype, dataspace;
   const static int RANK = 2;
   hsize_t dimsf[RANK];
   herr_t status;

   long stride;
   double disp;
   char hess_prefix[1024];

   int natoms;
   double ***local_hessian, ***hessian;
};

};

#endif
#endif
