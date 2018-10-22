#include "fix_nma.h"

#include <string.h>

#include "atom.h"
#include "comm.h"
#include "memory.h"
#include "input.h"
#include "modify.h"
#include "compute.h"
#include "update.h"
#include "force.h"
#include "lmptype.h"

#include "fix_evb.h"
#include "EVB_engine.h"

using namespace LAMMPS_NS;

FixNMA::FixNMA(LAMMPS *lmp, int narg, char **arg) :
   Fix(lmp, narg, arg),
   evoked(false),
   stride(2),
   disp(0.005),
   hess_prefix("hess"),
   natoms(0),
   local_hessian(NULL),
   hessian(NULL)
{
   // TODO parse fix to get stride and disp

   natoms = atom->natoms;
   dimsf[0] = dimsf[1] = 3 * natoms;
   if(!comm->me) {
      dataspace = H5Screate_simple(RANK, dimsf, NULL);
      datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
      status = H5Tset_order(datatype, H5T_ORDER_LE);
   }

   memory->create(local_hessian, 3 * natoms, natoms, 3, "hess");
   memory->create(hessian, 3 * natoms, natoms, 3, "hess");
}

FixNMA::~FixNMA()
{
   if(!comm->me) {
      H5Sclose(dataspace);
      H5Tclose(datatype);
   }
   memory->destroy(local_hessian);
   memory->destroy(hessian);
}

int FixNMA::setmask()
{
   int mask = 0;
   mask |= FixConst::PRE_FORCE;
   return mask;
}

void FixNMA::init()
{
}

void FixNMA::setup(int vflag) {
   if(strcmp(update->integrate_style, "verlet") == 0)
      pre_force(vflag);
}

void FixNMA::pre_force(int vflag) {

   if(evoked) return;

   if(update->ntimestep % stride) return;

   int fix_evb_id = modify->find_fix("1");
   FixEVB* fix_evb = static_cast<FixEVB*>(modify->fix[fix_evb_id]);

   double **fplus = NULL, **fminu = NULL;
   int nlocal = atom->nlocal;
   memory->create(fplus, nlocal, 3, "fplus");
   memory->create(fminu, nlocal, 3, "fminu");

//   for(int do_rank = 0; do_rank < comm->nprocs; ++do_rank) {
//
//      int nlocal_do_rank = 0;
//      if(comm->me == do_rank) nlocal_do_rank = nlocal;
//      MPI_Bcast(&nlocal_do_rank, 1, MPI_INT, do_rank, MPI_COMM_WORLD);
//
//      for(int ilocal = 0; ilocal < nlocal_do_rank; ++ilocal) {
//      }
//   }

   {
   int ilocal = 0;
   int dim = 0;
   int do_rank = 0;


   int itag = 0;
   if(comm->me == do_rank) itag = atom->tag[ilocal];
   MPI_Bcast(&itag, 1, MPI_INT, do_rank, MPI_COMM_WORLD);


   evoked = true;

   if(comm->me == do_rank) {
      atom->x[ilocal][dim] += disp;
   }
   MPI_Barrier(MPI_COMM_WORLD);
// @@@@
printf("plus barrier passed\n");
//   modify->pre_force(vflag);
//   modify->fix[fix_evb_id]->pre_force(vflag);
   fix_evb->Engine->execute(vflag);
   memcpy(&(fplus[0][0]), &(atom->f[0][0]), sizeof(double) * nlocal * 3);

// @@@@
printf("plus preforce passed\n");

   if(comm->me == do_rank) {
      atom->x[ilocal][dim] -= 2 * disp;
   }
   MPI_Barrier(MPI_COMM_WORLD);
//   modify->pre_force(vflag);
   fix_evb->Engine->execute(vflag);
   memcpy(&(fminu[0][0]), &(atom->f[0][0]), sizeof(double) * nlocal * 3);

// @@@@
printf("minus preforce passed\n");

   // recover original coordinates
   if(comm->me == do_rank) {
      atom->x[ilocal][dim] += disp;
   }

   // collect the hessian
   double **row = local_hessian[3 * (itag - 1) + dim];
   for(int jlocal = 0; jlocal < nlocal; ++jlocal) {
      unsigned id = atom->tag[jlocal] - 1;
      for(int dim = 0; dim < 3; ++dim) {
         row[id][dim] =
            (fplus[jlocal][dim] - fminu[jlocal][dim]) / 2.0 / disp;
      }
   }


   }

   memory->destroy(fminu);
   MPI_Reduce(&(local_hessian[0][0][0]), &(hessian[0][0][0]),
         9 * natoms * natoms, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

   // save the hessian to disk
   if(!comm->me) {
      double *data = &(hessian[0][0][0]);
      char hess_fname[1024];
      sprintf(hess_fname, "%s.%ld.h5", hess_prefix, update->ntimestep);
      hdf5_file =
      H5Fcreate(
            hess_fname,
            H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      dataset = H5Dcreate2(hdf5_file, "hess", datatype, dataspace,
                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
               H5P_DEFAULT, data);

      H5Dclose(dataset);
      H5Fclose(hdf5_file);
   }

   // this is the real pre_force
   modify->pre_force(vflag);

   evoked = false;
}

