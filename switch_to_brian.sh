grep -ilr 'import pyNN.nest' {*/*/*/,*/*/,*/,}*.py | xargs sed -i 's/import pyNN.nest/import pyNN.brian/g'
