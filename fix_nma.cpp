#include "fix_nma.h"

#include <string.h>
#include <math.h>

#include "atom.h"
#include "comm.h"
#include "memory.h"
#include "input.h"
#include "modify.h"
#include "compute.h"
#include "update.h"
#include "force.h"
#include "error.h"

using namespace LAMMPS_NS;

FixNMA::FixNMA(LAMMPS *lmp, int narg, char **arg) :
   Fix(lmp, narg, arg),
   fix_evb(NULL),
   stride(0),
   disp(0.0),
   hess_prefix("hess"),
   natoms(0),
   local_hessian(NULL),
   hessian(NULL)
{
   int next = 0;
   for(int i = 3; i < narg; ++i) {
      if(!strcmp(arg[i], "stride")) next = 1;
      else if(next == 1) {
        sscanf(arg[i], "%ld", &stride);
        next = 0;
      }
      else if(!strcmp(arg[i], "displacement")) next = 2;
      else if(next == 2) {
        sscanf(arg[i], "%lf", &disp);
        next = 0;
      }
      else if(!strcmp(arg[i], "outfile")) next = 3;
      else if(next == 3) {
        sscanf(arg[i], "%s", hess_prefix);
        next = 0;
      }
      else 
         error->all(FLERR,
         "syntax error in fix plumed - use "
         "'fix id all nma stride ... outfile ... displacement ...' ");
   }
   if(next==1) error->all(FLERR, "missing argument for stride option");
   if(next==2) error->all(FLERR, "missing argument for displacement option");

   if(stride <= 0) error->all(FLERR, "positive stride needed");
   if(fabs(disp) <= 1e-10) error->all(FLERR, "larger displacement needed");

   if(comm->me == 0 && screen) {
      fprintf(screen, "[NMA] Do NMA every %ld steps\n", stride);
      fprintf(screen, "[NMA] Use displacement %lf\n", disp);
      fprintf(screen, "[NMA] Save hessian with prefix %s\n", hess_prefix);
   }
   

   natoms = atom->natoms;
   dimsf[0] = dimsf[1] = 3 * natoms;
   if(!comm->me) {
      dataspace = H5Screate_simple(RANK, dimsf, NULL);
      datatype = H5Tcopy(H5T_NATIVE_DOUBLE);
      status = H5Tset_order(datatype, H5T_ORDER_LE);
   }

   memory->create(local_hessian, 3 * natoms, natoms, 3, "hess");
   memory->create(hessian, 3 * natoms, natoms, 3, "hess");

   EVB_GetFixObj(modify, &fix_evb);
   if(!fix_evb) 
      error->all(FLERR, "cannot find fix evb");
}

FixNMA::~FixNMA()
{
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


   // do not do this at the first timestep
   if(update->firststep == update->ntimestep) return;

   if(update->ntimestep % stride) return;


   double **fplus = NULL, **fminu = NULL;
   int nlocal = atom->nlocal;
   memory->create(fplus, nlocal, 3, "fplus");
   memory->create(fminu, nlocal, 3, "fminu");

   for(int do_rank = 0; do_rank < comm->nprocs; ++do_rank) {

      int nlocal_do_rank = 0;
      if(comm->me == do_rank) nlocal_do_rank = nlocal;
      MPI_Bcast(&nlocal_do_rank, 1, MPI_INT, do_rank, MPI_COMM_WORLD);

      for(int ilocal = 0; ilocal < nlocal_do_rank; ++ilocal) {

         int itag = 0;
         if(comm->me == do_rank) itag = atom->tag[ilocal];
         MPI_Bcast(&itag, 1, MPI_INT, do_rank, MPI_COMM_WORLD);
      
         for(int dim = 0; dim < 3; ++dim) {
      
            if(comm->me == do_rank) {
               atom->x[ilocal][dim] += disp;
            }
            memset(&(atom->f[0][0]), 0, sizeof(double)  * nlocal * 3);
            MPI_Barrier(MPI_COMM_WORLD);
            fix_evb->Engine->execute(vflag);
            if(force->newton) comm->reverse_comm();
            memcpy(&(fplus[0][0]), &(atom->f[0][0]), 
                  sizeof(double) * nlocal * 3);
      
            if(comm->me == do_rank) {
               atom->x[ilocal][dim] -= 2 * disp;
            }
            memset(&(atom->f[0][0]), 0, sizeof(double)  * nlocal * 3);
            MPI_Barrier(MPI_COMM_WORLD);
            fix_evb->Engine->execute(vflag);
            if(force->newton) comm->reverse_comm();
            memcpy(&(fminu[0][0]), &(atom->f[0][0]), 
                  sizeof(double) * nlocal * 3);
      
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



         } // end loop of dim


      } // end loop of ilocal
   } // end loop of do_rank

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
   memset(&(atom->f[0][0]), 0, sizeof(double)  * nlocal * 3);
   fix_evb->Engine->execute(vflag);

}

