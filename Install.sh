# Install/unInstall package files in LAMMPS

if (test $1 = 1) then

  if (test -e ../Makefile.package) then
    sed -i -e 's|^PKG_LIB =[ \t]*|& -lhdf5 |' ../Makefile.package
  fi
  cp fix_nma.cpp ..
  cp fix_nma.h ..

elif (test $1 = 0) then

  if (test -e ../Makefile.package) then
    sed -i -e 's/[^ \t]* -lhdf5[^ \t]* //' ../Makefile.package
  fi

  rm -f ../fix_nma.cpp
  rm -f ../fix_nma.h

fi
